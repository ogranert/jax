#include "jaxlib/mosaic/dialect/tpu/transforms/apply_vector_layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/Support/LogicalResult.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/transforms/apply_vector_layout_extensions.h"
#include "jaxlib/mosaic/dialect/tpu/transforms/infer_memref_layout.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"
#include "jaxlib/mosaic/dialect/tpu/vreg_util.h"
#include "xla/array.h"
#include "xla/layout.h"
#include "xla/util.h"

// TODO(tlongeri): Prefer returning failure over CHECKs. In particular, be more
// consistent about this for layout null checks in rules.

namespace mlir::tpu {
// TODO(tlongeri): Maybe just roll our own multi-dimensional array instead of
// using XLA's? There's too much glue for going from/to ArrayRef.

#define GEN_PASS_DECL_APPLYVECTORLAYOUTPASS
#define GEN_PASS_DEF_APPLYVECTORLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"


// The minimum bound required to rotate with scratch space. The bound refers to
// the number of VREGs on rotation dim. This number was concluded from some cost
// analysis for comparing different dynamic rotation implementations. If
// actual bound is greater than this, dynamic rotation with internal scratch
// space is more efficient.
// TODO(jevinjiang): need to update it based on the generation.
static constexpr int kMinBoundToRotateWithScratch = 27;

using RewriteContext = ApplyVectorLayoutContext;

LogicalResult applyLayoutBlock(RewriteContext &ctx, Block &block);
namespace {

void moveAllRegions(Operation &src, Operation &dst) {
  for (auto [src_region, dst_region] :
       llvm::zip_equal(src.getRegions(), dst.getRegions())) {
    dst_region.takeBody(src_region);
  }
}

// Get the address of pre-allocated internal scratch space with requested shape.
//
// Arguments:
//   shape: The shape of the requested scratch space.
//   elem_ty: The type of the elements in the requested scratch space.
//
// Returns:
//   A memref of the requested shape and type.
FailureOr<TypedValue<MemRefType>> getInternalScratch(
    RewriteContext &ctx, OpBuilder &builder, Location loc,
    ArrayRef<int64_t> shape, Type elem_ty, int64_t sublane_tiling = 0) {
  if (shape.empty()) {
    return failure();
  }
  if (shape.back() % ctx.target_shape[1] != 0) {
    return failure();
  }
  int packing = 32 / elem_ty.getIntOrFloatBitWidth();
  int sublane_count = llvm::divideCeil(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) /
          ctx.target_shape[1],
      packing);

  if (sublane_count > ctx.max_sublanes_in_scratch) {
    return failure();
  }
  // We can omit tpu_tiling_flags here because, for internal scratch, the
  // tiling does not matter (its shape is (N, 128)).
  FAILUREOR_ASSIGN_OR_RETURN(
      MemRefType scratch_ref_ty,
      inferMemref(MemRefType::get(shape, elem_ty), ctx.hardware_generation,
                  ctx.target_shape, /*tpu_tiling_flags=*/{}, sublane_tiling));
  return builder.create<tpu::GetInternalScratchOp>(loc, scratch_ref_ty)
      .getResult();
}

// Models Numpy's np.concatenate
xla::Array<Value> concatenate(const ArrayRef<xla::Array<Value>> arrays,
                              const int64_t axis) {
  CHECK(!arrays.empty());
  SmallVector<int64_t> dims(toArrayRef(arrays[0].dimensions()));
  CHECK(0 <= axis && axis < dims.size());
  for (size_t i = 1; i < arrays.size(); ++i) {
    CHECK_EQ(arrays[i].num_dimensions(), arrays[0].num_dimensions());
    for (size_t j = 0; j < arrays[i].num_dimensions(); ++j) {
      if (j != axis) {
        CHECK_EQ(arrays[i].dim(j), arrays[0].dim(j));
      }
    }
    dims[axis] += arrays[i].dim(axis);
  }
  xla::Array<Value> res(dims);
  int64_t offset = 0;
  for (xla::Array<Value> const& arr : arrays) {
    arr.Each([&](const absl::Span<const int64_t> idx, const Value v) {
      SmallVector<int64_t> res_idx(toArrayRef(idx));
      res_idx[axis] += offset;
      res(res_idx) = v;
    });
    offset += arr.dim(axis);
  }
  return res;
}

SmallVector<xla::Array<Value>> split(const xla::Array<Value> &vregs, int axis) {
  CHECK(axis >= 0 && axis < vregs.num_dimensions());
  SmallVector<xla::Array<Value>> chunks;
  chunks.reserve(vregs.dim(axis));
  SmallVector<int64_t> starts(vregs.num_dimensions(), 0);
  SmallVector<int64_t> limits(vregs.dimensions().begin(),
                              vregs.dimensions().end());
  for (int64_t i = 0; i < vregs.dim(axis); ++i) {
    starts[axis] = i;
    limits[axis] = i + 1;
    chunks.push_back(vregs.Slice(starts, limits));
  }
  return chunks;
};

template <typename T>
ArrayRef<T> XlaArrayToFlatArrayRef(xla::Array<T> xla_array) {
  return ArrayRef<T>(xla_array.data(), xla_array.num_elements());
}

template <typename T, typename Range>
xla::Array<T> XlaArrayFromShapeAndValues(ArrayRef<int64_t> sizes, Range vals) {
  // TODO(tlongeri): is there no way to avoid default initialization in the
  // constructor?
  xla::Array<T> arr(sizes);
  arr.SetValues(vals);
  return arr;
}

bool incrementSliceIndex(const MutableArrayRef<int64_t> idx,
                         const absl::Span<const int64_t> starts,
                         const absl::Span<const int64_t> limits) {
  const int64_t nd = idx.size();
  CHECK_EQ(nd, starts.size());
  CHECK_EQ(nd, limits.size());
  for (int64_t i = nd - 1; i >= 0; --i) {
    ++idx[i];
    if (idx[i] < limits[i]) {
      return true;
    }
    idx[i] = starts[i];
  }
  return false;
}

bool incrementIndex(const MutableArrayRef<int64_t> idx,
                    const absl::Span<const int64_t> limits) {
  const int64_t nd = idx.size();
  CHECK_EQ(nd, limits.size());
  for (int64_t i = nd - 1; i >= 0; --i) {
    ++idx[i];
    if (idx[i] < limits[i]) {
      return true;
    }
    idx[i] = 0;
  }
  return false;
}

bool sliceIsEmpty(const absl::Span<const int64_t> starts,
                  const absl::Span<const int64_t> limits) {
  for (auto [s, l] : llvm::zip_equal(starts, limits)) {
    CHECK_LE(s, l);
    if (s == l) {
      return true;
    }
  }
  return false;
}

// An alternative to xla::Array::UpdateSlice that takes a single value
template <typename T>
void updateSlice(xla::Array<T> &arr, const T &value,
                 const absl::Span<const int64_t> starts,
                 const absl::Span<const int64_t> limits) {
  if (sliceIsEmpty(starts, limits)) {
    return;
  }
  SmallVector<int64_t> idx(toArrayRef(starts));
  do {
    arr(idx) = value;
  } while (incrementSliceIndex(idx, starts, limits));
}

// An alternative to xla::Array::UpdateSlice that takes a range of data
template <typename T, typename Range>
void updateSliceFromRange(xla::Array<T> &arr, Range data,
                          const absl::Span<const int64_t> starts,
                          const absl::Span<const int64_t> limits) {
  if (sliceIsEmpty(starts, limits)) {
    return;
  }
  SmallVector<int64_t> idx(toArrayRef(starts));
  auto in_bounds = [&] {
    for (int64_t i = 0; i < idx.size(); ++i) {
      if (idx[i] >= arr.dim(i)) {
        return false;
      }
    }
    return true;
  };
  auto data_it = data.begin();
  do {
    if (in_bounds()) {
      arr(idx) = *data_it;
    }
    ++data_it;
  } while (incrementSliceIndex(idx, starts, limits));
  CHECK(data_it == data.end());
}

FailureOr<int64_t> getIntConst(Value v, bool silent = false) {
  if (auto constant_op = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer_attr = dyn_cast<IntegerAttr>(constant_op.getValue())) {
      return integer_attr.getValue().getSExtValue();
    }
  }
  if (silent) {
    return failure();
  }
  return emitError(v.getLoc(), "Expected an integer constant");
}

FailureOr<SmallVector<int64_t>> getIntConstsFromOperandRange(
    ValueRange vals, bool silent = false) {
  SmallVector<int64_t> res(vals.size());
  for (int i = 0; i < vals.size(); ++i) {
    FAILUREOR_ASSIGN_OR_RETURN(res[i], getIntConst(vals[i], silent));
  }
  return res;
}

Value broadcastSublane(OpBuilder &builder, Value vreg, int sublane_idx,
                       const std::array<int64_t, 2> target_shape) {
  return builder.create<tpu::GatherOp>(
      vreg.getLoc(), vreg.getType(), vreg,
      SmallVector<int32_t>(target_shape[0], sublane_idx),
      /*dimension=*/0);
}

FailureOr<std::pair<Value, SmallVector<int64_t>>> sliceRef(
    ImplicitLocOpBuilder &builder, TypedValue<MemRefType> base_ref,
    ArrayRef<int64_t> slice_shape, ValueRange indices,
    ArrayRef<int64_t> tiling) {
  IntegerType i32 = builder.getI32Type();
  MemRefType ref_ty = base_ref.getType();

  // MemRefSliceOp only allows tile-aligned slices. We pad the shape up
  // accordingly with the padding. We don't include the static tiled indices
  // in the slice when they can be arbitrary. But we do include dynamic tiled
  // indices under the condition that they are divisible by the tile size.
  SmallVector<int64_t> pad_slice_shape(slice_shape);
  TPU_ASSERT_LE_LOC(builder.getLoc(), tiling.size(), slice_shape.size());
  for (int i = 1; i <= tiling.size(); ++i) {
    auto &dim = *(pad_slice_shape.end() - i);
    dim = xla::RoundUpTo(dim, *(tiling.end() - i));
  }

  SmallVector<Value> slice_base_indices;
  slice_base_indices.reserve(ref_ty.getRank());
  for (auto idx : indices.drop_back(tiling.size())) {
    slice_base_indices.push_back(builder.create<arith::IndexCastOp>(i32, idx));
  }

  Value c0 = nullptr;
  SmallVector<int64_t> indices_within_slice(indices.size() - tiling.size(), 0);
  for (auto tiled_idx : indices.take_back(tiling.size())) {
    if (auto cst = getIntConst(tiled_idx, /*silent=*/true); succeeded(cst)) {
      indices_within_slice.push_back(*cst);
      if (!c0) {
        c0 = builder.create<arith::ConstantOp>(i32,
                                               builder.getI32IntegerAttr(0));
      }
      slice_base_indices.push_back(c0);
    } else {
      indices_within_slice.push_back(0);
      // TODO: Check divisibility!
      slice_base_indices.push_back(
          builder.create<arith::IndexCastOp>(i32, tiled_idx));
    }
  }

  // TODO(apaszke): Allow tile-aligned dynamic slicing on tiled dimensions.
  Value sliced_ref = builder.create<tpu::MemRefSliceOp>(
      MemRefType::get(pad_slice_shape, ref_ty.getElementType(),
                      ref_ty.getLayout(), ref_ty.getMemorySpace()),
      base_ref, slice_base_indices, /*dynamic_sizes=*/ValueRange());

  return std::make_pair(sliced_ref, indices_within_slice);
}

// Returns the first-level tiling of a (packed and tiled) memref value.
FailureOr<std::array<int64_t, 2>> getMemRefTiling(
    TypedValue<MemRefType> value, const std::array<int64_t, 2> target_shape) {
  if (auto erase_layout_op =
          dyn_cast_if_present<EraseLayoutOp>(value.getDefiningOp())) {
    value = erase_layout_op.getOperand();
  }
  const MemRefType memref_ty = value.getType();
  const auto mem_layout = dyn_cast<TiledLayoutAttr>(memref_ty.getLayout());
  if (mem_layout == nullptr) {
    return emitError(value.getLoc(), "Expected a tiled memref");
  }
  FAILUREOR_ASSIGN_OR_RETURN(int8_t bitwidth,
                             getTypeBitwidth(memref_ty.getElementType()));
  const int packing = 32 / bitwidth;
  const ArrayRef<xla::Tile> tiles = mem_layout.getTiles();
  const xla::Tile &first_tile = tiles.front();
  if (first_tile.dimensions().size() == 1) {
    const int64_t tile_size = first_tile.dimension(0);
    if (tile_size % (target_shape[1] * packing) != 0) {
      return emitError(value.getLoc(), "Not implemented");
    }
    if (bitwidth == 32) {
      if (tiles.size() > 1) {
        return emitError(value.getLoc(), "Not implemented");
      }
    } else if (bitwidth < 32) {
      if (tiles.drop_front() !=
          ArrayRef<xla::Tile>{xla::Tile({target_shape[1]}),
                              xla::Tile({packing, 1})}) {
        return emitError(value.getLoc(), "Not implemented");
      }
    }
    return std::array<int64_t, 2>{1, tile_size};
  }
  if (first_tile.dimensions().size() == 2) {
    if (bitwidth == 32) {
      if (tiles.size() > 1) {
        return emitError(value.getLoc(), "Not implemented");
      }
      return std::array<int64_t, 2>{first_tile.dimension(0),
                                    first_tile.dimension(1)};
    }
    if (bitwidth < 32) {
      if (tiles.size() != 2 || tiles[1] != xla::Tile({packing, 1})) {
        return emitError(value.getLoc(), "Not implemented");
      }
      return std::array<int64_t, 2>{first_tile.dimension(0),
                                    first_tile.dimension(1)};
    }
  }
  return emitError(value.getLoc(), "Not implemented");
}

// Hoist a vector constant as an additional argument of the function.
FailureOr<BlockArgument> appendConstant(RewriteContext &ctx, func::FuncOp func,
                                        DenseElementsAttr value) {
  MLIRContext *mlir_ctx = func.getContext();
  Block &entry_block = func.getBody().front();
  auto value_ty = cast<VectorType>(value.getType());
  if (value_ty.getElementType().getIntOrFloatBitWidth() != 32) {
    return func.emitOpError("Not implemented: Only 32-bit constants supported");
  }
  if (func->getAttr("scratch_operands")) {
    return func.emitOpError("Not implemented: function has scratch_operands");
  }
  // We can omit tpu_tiling_flags here since we invoke inferMemref only for
  // constant operands which are kernel parameters that will have their layouts
  // overridden before the pass pipeline runs anyway.
  FAILUREOR_ASSIGN_OR_RETURN(
      MemRefType arg_type,
      inferMemref(
          MemRefType::get(value_ty.getShape(), value_ty.getElementType()),
          ctx.hardware_generation, ctx.target_shape, /*tpu_tiling_flags=*/{},
          /*is_kernel_argument=*/true));
  const BlockArgument argument = entry_block.insertArgument(
      entry_block.getNumArguments() - 1, arg_type, UnknownLoc::get(mlir_ctx));
  const FunctionType func_ty = func.getFunctionType();
  // Adjust the function type.
  SmallVector<Type> new_arg_tys(func_ty.getInputs());
  new_arg_tys.insert(new_arg_tys.begin() + (new_arg_tys.size() - 1), arg_type);
  const auto new_func_ty =
      FunctionType::get(mlir_ctx, new_arg_tys, func_ty.getResults());
  func.setFunctionType(new_func_ty);
  // Adjust the constants attribute.
  if (auto prev_cst = func->getAttrOfType<ArrayAttr>("vector_constants")) {
    SmallVector<Attribute> vector_constants(prev_cst.getValue());
    vector_constants.push_back(value);
    func->setAttr("vector_constants",
                  ArrayAttr::get(func.getContext(), vector_constants));
  } else {
    func->setAttr("vector_constants", ArrayAttr::get(func.getContext(), value));
  }
  // Adjust window params for the extra operand.
  if (auto window_params = func->getAttrOfType<ArrayAttr>("window_params")) {
    const auto iteration_bounds =
        func->getAttrOfType<DenseI64ArrayAttr>("iteration_bounds");
    TPU_ASSERT_LOC(UnknownLoc::get(mlir_ctx), iteration_bounds);
    const int64_t iteration_rank = iteration_bounds.getSize();
    const SmallVector<AffineExpr> zeros(
        iteration_rank, getAffineConstantExpr(0, func.getContext()));
    const auto transform_indices =
        AffineMap::get(iteration_rank, 0, zeros, func.getContext());
    const auto new_param = DictionaryAttr::get(
        func.getContext(),
        NamedAttribute(StringAttr::get(func.getContext(), "transform_indices"),
                       AffineMapAttr::get(transform_indices)));
    SmallVector<Attribute> window_params_values(window_params.getValue());
    window_params_values.insert(window_params_values.end() - 1, new_param);
    func->setAttr("window_params",
                  ArrayAttr::get(func.getContext(), window_params_values));
  }
  return argument;
}

// Masks all values outside of bounds.
//
// Arguments:
//   value: A rank 2 MLIR vector to be masked.
//   bounds: A TargetTuple of slices specifying a rectangular subregion of value
//     that should be preserved during masking.
//   neutral: A scalar attribute specifying the value that will be inserted
//     for all values outside of specified bounds.
//
// Returns:
//   An MLIR value of the same type as the value argument, with all entries
//   outside of bounds replaced by neutral.
FailureOr<Value> maskOOB(RewriteContext &ctx, ImplicitLocOpBuilder &builder,
                         TypedValue<VectorType> value,
                         const VRegDataBounds &bounds,
                         const Attribute neutral) {
  auto native_vreg_ty =
      getNativeVregType(value.getType().getElementType(), ctx.target_shape);
  TPU_ASSERT_LOC(value.getLoc(), llvm::equal(value.getType().getShape(),
                                             native_vreg_ty.getShape()));
  if (bounds.isComplete(ctx.target_shape)) {
    return value;
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      TypedValue<VectorType> mask,
      bounds.getVectorMask(builder, value.getLoc(), ctx.hardware_generation,
                           ctx.target_shape));
  if (cast<IntegerType>(mask.getType().getElementType()).getWidth() != 1) {
    return emitError(value.getLoc(),
                     "Not implemented: Unsupported mask bitwidth");
  }
  if (mask.getType().getShape() != native_vreg_ty.getShape()) {
    mask = builder.create<tpu::MaskCastOp>(
        value.getLoc(),
        VectorType::get(native_vreg_ty.getShape(), builder.getI1Type()), mask);
  }
  Value neutral_vec = getFullVector(builder, native_vreg_ty, neutral);
  return builder
      .create<arith::SelectOp>(value.getLoc(), mask, value, neutral_vec)
      .getResult();
}

// Insert a minor dimension to the implicit shape. The original minor dimension
// becomes the new second minor dimension, laid out across sublanes.
//
// The returned vreg array uses the original tiling and the offsets specified in
// new_offsets to hold the value with the new implicit shape.
//
// Args:
// vregs:       The vreg array with *implicit* array shape.
// ishape:      The implicit shape of the represented value.
// layout:      The layout used for the represented value. The implicit
//              dimension is ignored, since this function operates directly at
//              the level of the implicit shape.
// new_offsets: The offsets to use for the layout of the returned vreg array.
FailureOr<xla::Array<Value>> insertImplicitMinorDimension(
    RewriteContext &ctx, OpBuilder &builder, const Location loc,
    const xla::Array<Value> &vregs, const ArrayRef<int64_t> ishape,
    const VectorLayout &layout, const LayoutOffsets new_offsets) {
  if (layout.bitwidth() != 32 || !layout.hasNativeTiling(ctx.target_shape)) {
    return emitError(loc, "Not implemented: Unsupported bitwidth or tiling");
  }
  if (layout.offsets()[1].has_value()) {
    if (!new_offsets[0]) {
      // TODO(tlongeri): This can only be valid if the dim size is 1.
      return emitError(loc, "Not implemented: Replication mismatch");
    }
    if (*new_offsets[0] != *layout.offsets()[1] % ctx.target_shape[0] &&
        *layout.offsets()[1] + *(ishape.end() - 1) > ctx.target_shape[1]) {
      // This requires blending data from different vregs.
      return emitError(loc,
                       "Not implemented: Misaligned offsets and shape does not "
                       "fit in one vreg");
    }
  }
  // new_layout is only to get the new vreg array shape, the implicit dim is
  // irrelevant (since we already have the implicit shape):
  const VectorLayout new_layout(layout.bitwidth(), new_offsets, layout.tiling(),
                                VectorLayout::ImplicitDim::kNone);
  SmallVector<int64_t> new_ishape(ishape);
  new_ishape.push_back(1);
  xla::Array<Value> new_vregs(new_layout.tileArrayShape(
      /*src_is_implicit=*/true, /*res_is_implicit=*/true, std::move(new_ishape),
      ctx.target_shape));
  // Preallocate an indices vector to avoid repeated allocations:
  SmallVector<int64_t> idxs;
  new_vregs.Each([&](const absl::Span<const int64_t> dst_idx,
                     Value *const dst_vreg) {
    // Indices of the new vreg in the new vreg array:
    const int64_t new_2nd_minor_idx = *(dst_idx.end() - 2);
    const int64_t new_3rd_minor_idx = *(dst_idx.end() - 3);
    idxs.assign(dst_idx.begin(), dst_idx.end());
    if (!layout.offsets()[0].has_value() && new_3rd_minor_idx != 0) {
      // All vregs along that dimension are the same
      *(idxs.end() - 3) = 0;
      *dst_vreg = new_vregs(idxs);
    } else if (!layout.offsets()[1].has_value() && new_2nd_minor_idx != 0) {
      // All vregs along that dimension are the same
      *(idxs.end() - 2) = 0;
      *dst_vreg = new_vregs(idxs);
    } else {
      // dst_vreg will hold slice [row_idx, col_idx:(col_idx + target_shape[0])]
      // of the after-offsets source shape
      const int64_t row_idx =
          layout.offsets()[0] ? new_3rd_minor_idx + *layout.offsets()[0] : 0;
      const int64_t col_idx = layout.offsets()[1]
                                  ? new_2nd_minor_idx * ctx.target_shape[0] +
                                        *layout.offsets()[1] - *new_offsets[0]
                                  : 0;

      idxs.pop_back();
      *(idxs.end() - 2) = row_idx / ctx.target_shape[0];
      *(idxs.end() - 1) = col_idx / ctx.target_shape[1];
      Value src_vreg = vregs(idxs);
      // TODO(tlongeri): We can sometimes skip operations when dst_vreg will
      // hold a single non-padding element (first or last) and we don't need
      // replication in the output.
      if (layout.offsets()[0].has_value()) {
        // [ . . . . . . . . ]    [ . . . . a b c d ]
        // [ . . . . a b c d ] => [ . . . . a b c d ]
        // [ . . . . . . . . ]    [ . . . . a b c d ]
        // [ . . . . . . . . ]    [ . . . . a b c d ]
        src_vreg = broadcastSublane(
            builder, src_vreg,
            /*sublane_idx=*/row_idx % ctx.target_shape[0], ctx.target_shape);
      }
      if (layout.offsets()[1].has_value()) {
        // [ . . . . a b c d ]    [ a a a a a a a a ]
        // [ . . . . a b c d ] => [ b b b b b b b b ]
        // [ . . . . a b c d ]    [ c c c c c c c c ]
        // [ . . . . a b c d ]    [ d d d d d d d d ]
        src_vreg = builder.create<BroadcastInSublanesOp>(
            loc, src_vreg.getType(), src_vreg,
            /*lane=*/col_idx % ctx.target_shape[1]);
      }
      *dst_vreg = src_vreg;
    }
  });
  return new_vregs;
}

LogicalResult elementwise_op_rule(RewriteContext &ctx, Operation &op,
                                  const ArrayRef<Layout> layouts_in,
                                  const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_OP(OpTrait::hasElementwiseMappableTraits(&op));
  if (op.getNumResults() != 1) {
    return op.emitError("Not implemented: Only ops with one result supported");
  }
  TPU_ASSERT_EQ_OP(layouts_in.size(), op.getNumOperands());
  TPU_ASSERT_GT_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  OpBuilder builder(&op);
  if (!(layouts_out.front().has_value() &&
        llvm::all_of(layouts_in,
                     [&](const Layout &l) { return l.has_value(); }))) {
    return op.emitOpError(
        "Not implemented: Null layout / non-vector operand in elementwise "
        "operation");
  }
  const auto out_ty = cast<VectorType>(op.getResult(0).getType());
  const VectorLayout &layout_out = *layouts_out.front();
  if (!llvm::all_of(layouts_in, [&](const Layout &l) {
        return l->generalizes(layout_out, out_ty.getShape(), ctx.target_shape);
      })) {
    return op.emitOpError(
        "Not implemented: Incompatible layouts in elementwise operation");
  }
  const unsigned num_operands = op.getNumOperands();
  SmallVector<xla::Array<Value>> in_vreg_arrays;
  in_vreg_arrays.reserve(num_operands);
  for (unsigned i = 0; i < num_operands; ++i) {
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> tile_array,
        disassemble(builder, *layouts_in[i],
                    cast<TypedValue<VectorType>>(op.getOperand(i)),
                    ctx.target_shape));
    in_vreg_arrays.emplace_back(std::move(tile_array));
  }

  const VectorType out_vreg_ty = getNativeVregOrVmaskType(
      out_ty.getElementType(), layout_out.bitwidth(), ctx.target_shape);

  NamedAttrList attributes(op.getAttrDictionary());
  attributes.erase("in_layout");
  attributes.erase("out_layout");

  // Note that we have to broadcast to handle replicate dimensions.
  SmallVector<int64_t> broadcasted_shape(
      toArrayRef(in_vreg_arrays[0].dimensions()));
  for (size_t i = 1; i < num_operands; ++i) {
    SmallVector<int64_t> new_broadcasted_shape;
    TPU_ASSERT_OP(OpTrait::util::getBroadcastedShape(
        broadcasted_shape, toArrayRef(in_vreg_arrays[i].dimensions()),
        new_broadcasted_shape));
    broadcasted_shape = std::move(new_broadcasted_shape);
  }
  TPU_ASSERT_OP(broadcasted_shape ==
                layout_out.tileArrayShape(out_ty.getShape(), ctx.target_shape));

  // TODO(tlongeri): Can we avoid initializing the array before filling values?
  xla::Array<Value> out_vreg_array(broadcasted_shape);
  out_vreg_array.Each([&](absl::Span<const int64_t> idx, Value *out_vreg) {
    SmallVector<Value> operands(num_operands);

    for (unsigned i = 0; i < num_operands; ++i) {
      // Handle indices for broadcasted dimensions
      SmallVector<int64_t> operand_idx(toArrayRef(idx));
      for (unsigned j = 0; j < idx.size(); ++j) {
        if (in_vreg_arrays[i].dim(j) == 1) {
          operand_idx[j] = 0;
        }
      }
      operands[i] = in_vreg_arrays[i](operand_idx);
    }
    Operation *vreg_op =
        builder.create(op.getLoc(), op.getName().getIdentifier(), operands,
                       out_vreg_ty, attributes.getAttrs());
    CHECK(vreg_op);
    CHECK_EQ(vreg_op->getNumResults(), 1);
    *out_vreg = vreg_op->getResult(0);
  });
  op.replaceAllUsesWith(assemble(builder, out_ty, layout_out,
                                 std::move(out_vreg_array), ctx.target_shape));
  op.erase();
  return success();
}

using rule_type = std::function<LogicalResult(
    RewriteContext &, Operation &, ArrayRef<Layout>, ArrayRef<Layout>)>;

template <typename OpTy>
FailureOr<xla::Array<Value>> ext_op_rule_impl(RewriteContext &ctx,
                                              OpBuilder &builder, OpTy op,
                                              const VectorLayout &layout_in,
                                              const VectorLayout &layout_out) {
  const auto result_ty = cast<VectorType>(op.getResult().getType());
  auto source = cast<TypedValue<VectorType>>(op.getIn());
  auto output_vregs_shape =
      layout_out.tileArrayImplicitShape(result_ty.getShape(), ctx.target_shape);
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> input_vregs,
      disassemble(builder, layout_in, source, ctx.target_shape,
                  /*use_implicit_shape=*/true));
  xla::Array<Value> output_vregs(output_vregs_shape);
  const VectorType res_vreg_ty =
      getNativeVregType(result_ty.getElementType(), ctx.target_shape);
  if (layout_in.implicit_dim() != layout_out.implicit_dim()) {
    return op.emitOpError(
        "Not implemented: Change of implicit dim during the cast");
  }
  if (layout_in.offsets() != layout_out.offsets()) {
    return op.emitOpError("Not implemented: Change of offsets during the cast");
  }
  const int packing = layout_out.bitwidth() / layout_in.bitwidth();
  if (layout_in.hasNativeTiling(ctx.target_shape) &&
      layout_out.hasNativeTiling(ctx.target_shape)) {
    output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      SmallVector<int64_t> input_vreg_idxs(toArrayRef(idxs));
      int64_t vreg_part = *(input_vreg_idxs.end() - 2) % packing;
      *(input_vreg_idxs.end() - 2) /= packing;
      *v = builder.create<UnpackSubelementsOp>(
          op.getLoc(), res_vreg_ty, input_vregs(input_vreg_idxs), vreg_part,
          tpu::PackFormat::kCompressed);
    });
  } else {
    if (layout_in.tiling() != layout_out.tiling()) {
      return op.emitOpError("Not implemented: Changing tiling during the cast");
    }
    auto tiling = layout_in.tiling();
    if (ctx.target_shape[0] % tiling[0] != 0 ||
        ctx.target_shape[1] != tiling[1]) {
      return op.emitOpError("Not implemented: tiling not supported");
    }
    output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      SmallVector<int64_t> input_vreg_idxs(toArrayRef(idxs));
      input_vreg_idxs.back() /= packing;
      const int64_t vreg_part = idxs.back() % packing;
      *v = builder.create<UnpackSubelementsOp>(
          op.getLoc(), res_vreg_ty, input_vregs(input_vreg_idxs), vreg_part,
          tpu::PackFormat::kCompressed);
    });
  }
  return output_vregs;
}

LogicalResult arith_extf_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto extf_op = cast<arith::ExtFOp>(op);
  if (layouts_in.front()->bitwidth() != 16 ||
      layouts_out.front()->bitwidth() != 32) {
    return op.emitOpError(
        "Not implemented: Only 16-bit to 32-bit conversion supported");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> output_vregs,
      ext_op_rule_impl(ctx, builder, extf_op, *layouts_in.front(),
                       *layouts_out.front()));
  const auto result_ty = cast<VectorType>(extf_op.getResult().getType());
  extf_op.replaceAllUsesWith(assemble(builder, result_ty, *layouts_out.front(),
                                      std::move(output_vregs), ctx.target_shape,
                                      /*use_implicit_shape=*/true)
                                 .getResult());
  extf_op.erase();
  return success();
}

LogicalResult arith_extsi_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto extsi_op = cast<arith::ExtSIOp>(op);
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> output_vregs,
      ext_op_rule_impl(ctx, builder, extsi_op, *layouts_in.front(),
                       *layouts_out.front()));
  const auto result_ty = cast<VectorType>(extsi_op.getResult().getType());
  extsi_op.replaceAllUsesWith(assemble(builder, result_ty, *layouts_out.front(),
                                       std::move(output_vregs),
                                       ctx.target_shape,
                                       /*use_implicit_shape=*/true)
                                  .getResult());
  extsi_op.erase();
  return success();
}

LogicalResult arith_extui_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto extui_op = cast<arith::ExtUIOp>(op);
  const auto in_ty = cast<VectorType>(extui_op.getIn().getType());
  const auto out_ty = cast<VectorType>(extui_op.getType());
  const unsigned in_bitwidth = in_ty.getElementTypeBitWidth();
  if (in_bitwidth == 1) {
    return elementwise_op_rule(ctx, op, layouts_in, layouts_out);
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> output_vregs,
      ext_op_rule_impl(ctx, builder, extui_op, *layouts_in.front(),
                       *layouts_out.front()));
  unsigned out_bitwidth = out_ty.getElementTypeBitWidth();
  // Generate a mask to mask out the sign extension. e.g., for u8 -> u16,
  // the mask is 0x00ff00ff.
  unsigned mask = (1 << in_bitwidth) - 1;
  while (out_bitwidth < 32) {
    mask = (mask << out_bitwidth) | mask;
    out_bitwidth *= 2;
  }
  const VectorType i32_vreg_ty =
      getNativeVregType(builder.getI32Type(), ctx.target_shape);
  auto mask_const = builder.create<arith::ConstantOp>(
      op.getLoc(), i32_vreg_ty, DenseIntElementsAttr::get(i32_vreg_ty, {mask}));
  const VectorType out_vreg_ty =
      getNativeVregType(out_ty.getElementType(), ctx.target_shape);
  output_vregs.Each([&](absl::Span<const int64_t> _, Value *v) {
    Value unpacked =
        builder.create<BitcastVregOp>(op.getLoc(), i32_vreg_ty, *v);
    unpacked = builder.create<arith::AndIOp>(op.getLoc(), i32_vreg_ty, unpacked,
                                             mask_const);
    *v = builder.create<BitcastVregOp>(op.getLoc(), out_vreg_ty, unpacked);
  });
  extui_op.replaceAllUsesWith(assemble(builder, out_ty, *layouts_out.front(),
                                       std::move(output_vregs),
                                       ctx.target_shape,
                                       /*use_implicit_shape=*/true)
                                  .getResult());
  extui_op.erase();
  return success();
}

template <typename OpTy>
LogicalResult trunc_op_rule_impl(RewriteContext &ctx, OpTy op,
                                 const VectorLayout &layout_in,
                                 const VectorLayout &layout_out) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());
  auto source = cast<TypedValue<VectorType>>(op.getIn());
  auto result_ty = cast<VectorType>(op.getResult().getType());
  auto output_vregs_shape =
      layout_out.tileArrayImplicitShape(result_ty.getShape(), ctx.target_shape);
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> input_vregs,
      disassemble(builder, layout_in, source, ctx.target_shape,
                  /*use_implicit_shape=*/true));
  xla::Array<Value> output_vregs(output_vregs_shape);
  const LayoutOffsets input_offsets = layout_in.offsets();
  const LayoutOffsets output_offsets = layout_out.offsets();
  const std::array<int64_t, 2> input_vreg_slice =
      layout_in.vregSlice(ctx.target_shape);
  const std::array<int64_t, 2> output_vreg_slice =
      layout_out.vregSlice(ctx.target_shape);
  const int input_sublanes_per_tile =
      layout_in.sublanesPerTile(ctx.target_shape);

  if (layout_in.implicit_dim() != layout_out.implicit_dim()) {
    return op.emitOpError(
        "Not implemented: Truncation changes implicit dimension");
  }
  for (const auto &[input_offset, output_offset, input_slice_size] :
       llvm::zip_equal(input_offsets, output_offsets, input_vreg_slice)) {
    if (!input_offset.has_value() && !output_offset.has_value()) {
      // Replicated to replicated is okay
    } else if (!input_offset.has_value() && output_offset.has_value()) {
      // Replicated to non-replicated could be handled, but we don't leverage
      // replication, so we don't expect a replicated input offset to be
      // assigned. The materialization of replicated vregs in the vreg
      // array should be handled by relayout.
      return op.emitOpError(
          "Not implemented: Replicated to non-replicated offset");
    } else if (input_offset.has_value() && !output_offset.has_value()) {
      return op.emitOpError(
          "Not implemented: Truncation introduces replication");
    } else {
      DCHECK(input_offset.has_value() && output_offset.has_value());
      if (*input_offset != *output_offset % input_slice_size) {
        return op.emitOpError("Not implemented: Misaligned offsets");
      }
    }
  }
  if (output_vreg_slice[0] % input_vreg_slice[0] != 0 ||
      output_vreg_slice[1] % input_vreg_slice[1] != 0) {
    // The output vreg slice should be a union of whole input vreg slices
    return op.emitOpError("Not implemented: Unsupported tiling change");
  }
  // How many rows and columns of input vregs we are packing into one output
  // vreg:
  const int64_t vreg_rows = output_vreg_slice[0] / input_vreg_slice[0];
  const int64_t vreg_cols = output_vreg_slice[1] / input_vreg_slice[1];

  // Currently, we always pack across rows first, and then across columns.
  // Note: Even though we combine it into a single tpu.pack_subelements op, the
  //       order of the operands is such that it is equivalent to packing across
  //       rows and then across columns.
  // TODO(b/384274392): For some cases we want to pack across columns first, but
  //                    we also need mixed compressed/interleaved packing.

  // The format for packing *across* multiple rows in the vreg array (different
  // 2nd minor index):
  PackFormat row_pack_format = PackFormat::kCompressed;
  if (vreg_rows != 1) {
    // When going from (a, b) to (a * n, b) tiling, each output tile is the
    // union of n input tiles from different vregs. The ith tile of the output
    // vreg is formed by packing the ith tiles of the input vregs together.
    // This can only be done when tiles are one sublane (by packing interleaved)
    // or when they occupy the full vreg (by packing compressed).
    // Note: Currently, we always pack across rows before packing across
    //       columns, so we just check the source tiling.
    if (input_sublanes_per_tile == 1) {
      row_pack_format = PackFormat::kInterleaved;
    } else if (input_sublanes_per_tile == ctx.target_shape[0]) {
      row_pack_format = PackFormat::kCompressed;
    } else {
      return op.emitOpError(
          "Not implemented: Tiling change requires interleaving tiles that are "
          "not one sublane or one full vreg");
    }
  }
  // The tiling after packing across rows:
  const std::array<int64_t, 2> intermediate_tiling = {
      layout_in.tiling()[0] * vreg_rows, layout_in.tiling()[1]};
  DCHECK_EQ(intermediate_tiling[0], layout_out.tiling()[0]);

  // We only support compressed packing across vreg columns, which doesn't
  // change the tiling. Logically, it just stacks tiles horizontally.
  if (intermediate_tiling[1] != layout_out.tiling()[1] &&
      // For (1, x) tiling all minor dimension tilings are equivalent, although
      // some are illegal in VectorLayout. So, even though compressed packing in
      // general does not change the tiling, for (1, x) we can still change to
      // other minor dimension tilings (they are equivalent).
      intermediate_tiling[0] != 1) {
    // This could be handled, in some cases, by using interleaved packing across
    // vreg columns, but we never use tilings like this. An example where we
    // could use interleaved packing is (8, 128) f32 -> (8, 256) bf16.
    return op.emitOpError(
        "Not implemented: Truncating to increasing minor tile size");
  }
  // The format for packing *across* multiple columns in the vreg array
  // (different minor index):
  constexpr PackFormat col_pack_format = PackFormat::kCompressed;

  if (vreg_rows != 1 && vreg_cols != 1 && row_pack_format != col_pack_format) {
    // TODO(b/384274392): We can alternate interleaved and compressed packing
    //                    but how should we expose it in tpu.pack_subelements?
    return op.emitOpError(
        "Not implemented: Tiling change requires mixed compressed and "
        "interleaved packing");
  }
  const PackFormat pack_format =
      vreg_rows != 1 ? row_pack_format : col_pack_format;

  const VectorType res_vreg_ty =
      getNativeVregType(result_ty.getElementType(), ctx.target_shape);

  SmallVector<int64_t> input_idx;
  output_vregs.Each([&](absl::Span<const int64_t> output_idx, Value *v) {
    SmallVector<Value> parts;
    input_idx.assign(output_idx.begin(), output_idx.end());
    auto push_col = [&]() {
      if (!output_offsets[0].has_value()) {
        *(input_idx.end() - 2) = 0;
        // Make sure we set all rows of the column to make it replicated
        parts.append(vreg_rows, input_vregs(input_idx));
      } else {
        const int64_t row_offset = *output_offsets[0] / input_vreg_slice[0];
        const int64_t base_src_row =
            *(output_idx.end() - 2) * vreg_rows - row_offset;
        for (int64_t row = base_src_row; row < base_src_row + vreg_rows;
             ++row) {
          if (0 <= row && row < *(input_vregs.dimensions().end() - 2)) {
            *(input_idx.end() - 2) = row;
            parts.push_back(input_vregs(input_idx));
          } else {
            parts.push_back(nullptr);
          }
        }
      }
    };
    if (!output_offsets[1].has_value()) {
      *(input_idx.end() - 1) = 0;
      // Make sure we set all column parts of the vreg to make it replicated
      push_col();
      for (int64_t col = 1; col < vreg_cols; ++col) {
        for (int64_t row = 0; row < vreg_rows; ++row) {
          parts.push_back(parts[row]);
        }
      }
    } else {
      const int64_t col_offset = *output_offsets[1] / input_vreg_slice[1];
      const int64_t base_src_col =
          *(output_idx.end() - 1) * vreg_cols - col_offset;
      for (int64_t col = base_src_col; col < base_src_col + vreg_cols; ++col) {
        if (0 <= col && col < *(input_vregs.dimensions().end() - 1)) {
          *(input_idx.end() - 1) = col;
          push_col();
        } else {
          parts.append(vreg_rows, nullptr);
        }
      }
    }
    *v = builder.create<PackSubelementsOp>(res_vreg_ty, parts, pack_format);
  });
  op.replaceAllUsesWith(assemble(builder, result_ty, layout_out,
                                 std::move(output_vregs), ctx.target_shape,
                                 /*use_implicit_shape=*/true)
                            .getResult());
  op.erase();
  return success();
}

LogicalResult arith_truncf_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto truncf_op = cast<arith::TruncFOp>(op);
  if (layouts_in.front()->bitwidth() != 32 ||
      (layouts_out.front()->bitwidth() != 16 &&
       layouts_out.front()->bitwidth() != 8)) {
    return op.emitOpError(
        "Not implemented: Only 32-bit to 16-or-8-bit conversion supported");
  }
  return trunc_op_rule_impl(ctx, truncf_op, *layouts_in.front(),
                            *layouts_out.front());
}

LogicalResult arith_trunci_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto trunci_op = cast<arith::TruncIOp>(op);
  return trunc_op_rule_impl(ctx, trunci_op, *layouts_in.front(),
                            *layouts_out.front());
}

LogicalResult func_return_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_OP(layouts_out.empty());
  for (const Layout &layout_in : layouts_in) {
    if (layout_in.has_value()) {
      return op.emitOpError("Vector-typed return values are not supported");
    }
  }
  return success();
}

LogicalResult scf_for_rule(RewriteContext &ctx, Operation &op,
                           const ArrayRef<Layout> layouts_in,
                           const ArrayRef<Layout> layouts_out) {
  scf::ForOp for_op = cast<scf::ForOp>(op);
  TPU_ASSERT_EQ_OP(layouts_in.size(), for_op->getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), for_op->getNumResults());
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<Layout> yield_in_layouts,
      getInLayouts(*for_op.getBody()->getTerminator(), ctx.target_shape));
  int out_idx = 0;
  for (auto [in_layout, yield_layout, out_layout, result] :
       llvm::zip_equal(layouts_in.drop_front(3), yield_in_layouts, layouts_out,
                       op.getResults())) {
    if (auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(in_layout.has_value());
      TPU_ASSERT_OP(yield_layout.has_value());
      TPU_ASSERT_OP(out_layout.has_value());
      if (in_layout.value() != yield_layout.value()) {
        return op.emitOpError(
                   "Not implemented: for loop input layout does not match with "
                   "yield layout ")
               << out_idx;
      }
      if (in_layout.value() != out_layout.value()) {
        return op.emitOpError(
                   "Not implemented: for loop input layout does not match with "
                   "out layout ")
               << out_idx;
      }
    } else {
      TPU_ASSERT_EQ_OP(in_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(yield_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(out_layout, kNoLayout);
    }
    ++out_idx;
  }

  if (failed(applyLayoutBlock(ctx, *for_op.getBody()))) {
    return failure();
  }

  if (op.getNumResults() == 0) {
    return success();
  }

  OpBuilder builder(&op);
  SmallVector<Value> unrolled_args;
  for (int i = 0; i < layouts_in.size(); ++i) {
    auto layout = layouts_in[i];
    auto operand = for_op.getOperand(i);
    if (i < 3) {
      if (layout.has_value()) {
        return op.emitOpError("Expected no layout for bounds and step");
      }
      continue;
    }
    if (auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand)) {
      if (!layout.has_value()) {
        return op.emitOpError("Expected layout for vector operand");
      }
      FAILUREOR_ASSIGN_OR_RETURN(
          const xla::Array<Value> tiles,
          disassemble(builder, *layout, vector_operand, ctx.target_shape));
      unrolled_args.append(tiles.begin(), tiles.end());
    } else {
      if (layout.has_value()) {
        return op.emitOpError("Expected no layout for scalar operand");
      }
      unrolled_args.push_back(operand);
    }
  }

  // Create a new scf::ForOp with unrolled args.
  auto new_op = builder.create<scf::ForOp>(
      for_op->getLoc(), for_op.getLowerBound(), for_op.getUpperBound(),
      for_op.getStep(), unrolled_args);

  int num_old_args = for_op.getBody()->getNumArguments();
  SmallVector<Location> locs(new_op.getBody()->getNumArguments(),
                             for_op.getLoc());
  for_op.getBody()->addArguments(TypeRange(new_op.getBody()->getArguments()),
                                 locs);
  builder.setInsertionPointToStart(for_op.getBody());
  auto arg_idx = num_old_args;
  // Block also has an induction variable that should have no layout,
  // which conveniently matches the in layouts.
  for (auto [old_arg, layout] : llvm::zip_equal(
           for_op.getBody()->getArguments().take_front(num_old_args),
           layouts_in.drop_front(2))) {
    if (const auto vty = dyn_cast<VectorType>(old_arg.getType())) {
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(arg_idx + num_vectors,
                       for_op.getBody()->getNumArguments());
      tiles.SetValues(llvm::make_range(
          for_op.getBody()->getArguments().begin() + arg_idx,
          for_op.getBody()->getArguments().begin() + arg_idx + num_vectors));
      arg_idx += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      old_arg.replaceUsesWithIf(rolled_op, [&](OpOperand &operand) {
        return operand.getOwner() != rolled_op;
      });
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      old_arg.replaceAllUsesWith(for_op.getBody()->getArgument(arg_idx));
      ++arg_idx;
    }
  }
  for_op.getBody()->eraseArguments(0, num_old_args);
  new_op.getRegion().takeBody(for_op.getRegion());

  // Roll the results back to the original shapes.
  builder.setInsertionPointAfter(new_op);
  int64_t res_idx = 0;
  SmallVector<Value> rolled_results;
  for (auto [result, layout] :
       llvm::zip_equal(for_op.getResults(), layouts_out)) {
    if (const auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(res_idx + num_vectors, new_op.getResults().size());
      tiles.SetValues(llvm::make_range(
          new_op.getResults().begin() + res_idx,
          new_op.getResults().begin() + res_idx + num_vectors));
      res_idx += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      rolled_results.push_back(rolled_op);
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      rolled_results.push_back(new_op.getResult(res_idx));
      ++res_idx;
    }
  }

  for_op.replaceAllUsesWith(rolled_results);
  for_op.erase();
  return success();
}

LogicalResult scf_while_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  scf::WhileOp while_op = cast<scf::WhileOp>(op);
  TPU_ASSERT_EQ_OP(layouts_in.size(), while_op->getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), while_op->getNumResults());
  TPU_ASSERT_EQ_OP(layouts_in.size(), layouts_out.size());

  // The terminator for the before region is the condition op.
  // It takes multiple arguments -- the first being the decision to execute the
  // after region or branch to the exit.
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<Layout> cond_in_layouts,
      getInLayouts(*while_op.getBeforeBody()->getTerminator(),
                   ctx.target_shape));

  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<Layout> yield_in_layouts,
      getInLayouts(*while_op.getYieldOp(), ctx.target_shape));
  int out_idx = 0;
  for (auto [in_layout, cond_layout, yield_layout, out_layout, result] :
       llvm::zip_equal(layouts_in,
                       ArrayRef<Layout>(cond_in_layouts).drop_front(1),
                       yield_in_layouts, layouts_out, op.getResults())) {
    if (auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(in_layout.has_value());
      TPU_ASSERT_OP(yield_layout.has_value());
      TPU_ASSERT_OP(out_layout.has_value());
      if (in_layout.value() != cond_layout.value()) {
        return op.emitOpError(
                   "Not implemented: while loop input layout does not match "
                   "with condition layout ")
               << out_idx;
      }
      if (in_layout.value() != yield_layout.value()) {
        return op.emitOpError(
                   "Not implemented: while loop input layout does not match "
                   "with yield layout ")
               << out_idx;
      }
      if (in_layout.value() != out_layout.value()) {
        return op.emitOpError(
                   "Not implemented: while loop input layout does not match "
                   "with output layout ")
               << out_idx;
      }
    } else {
      TPU_ASSERT_EQ_OP(in_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(cond_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(yield_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(out_layout, kNoLayout);
    }
    ++out_idx;
  }

  if (failed(applyLayoutBlock(ctx, *while_op.getBeforeBody()))) {
    return failure();
  }

  if (failed(applyLayoutBlock(ctx, *while_op.getAfterBody()))) {
    return failure();
  }

  if (op.getNumResults() == 0) {
    return success();
  }

  OpBuilder builder(&op);
  SmallVector<Value> unrolled_args;
  for (int i = 0; i < layouts_in.size(); ++i) {
    auto layout = layouts_in[i];
    auto operand = while_op.getOperand(i);
    if (auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand)) {
      if (!layout.has_value()) {
        return op.emitOpError("Expected layout for vector operand");
      }
      FAILUREOR_ASSIGN_OR_RETURN(
          const xla::Array<Value> tiles,
          disassemble(builder, *layout, vector_operand, ctx.target_shape));
      unrolled_args.append(tiles.begin(), tiles.end());
    } else {
      if (layout.has_value()) {
        return op.emitOpError("Expected no layout for scalar operand");
      }
      unrolled_args.push_back(operand);
    }
  }

  // Create a new scf::WhileOp with unrolled args.
  auto new_op = builder.create<scf::WhileOp>(
      while_op->getLoc(),
      TypeRange(while_op.getConditionOp().getOperands().drop_front(1)),
      unrolled_args, nullptr, nullptr);

  const auto tile_body_args = [&](::mlir::Block *old_body,
                                  ::mlir::Block *new_body,
                                  const ArrayRef<Layout> layouts) {
    TPU_ASSERT_OP(old_body != nullptr);
    TPU_ASSERT_OP(new_body != nullptr);
    int num_old_args = old_body->getNumArguments();
    SmallVector<Location> locs(new_body->getNumArguments(), while_op.getLoc());
    old_body->addArguments(TypeRange(new_body->getArguments()), locs);
    builder.setInsertionPointToStart(old_body);
    auto arg_idx = num_old_args;
    for (auto [old_arg, layout] : llvm::zip_equal(
             old_body->getArguments().take_front(num_old_args), layouts)) {
      if (const auto vty = dyn_cast<VectorType>(old_arg.getType())) {
        TPU_ASSERT_OP(layout.has_value());
        const SmallVector<int64_t> tiles_shape =
            layout->tileArrayShape(vty.getShape(), ctx.target_shape);
        const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
        xla::Array<Value> tiles(tiles_shape);
        TPU_ASSERT_LE_OP(arg_idx + num_vectors, old_body->getNumArguments());
        tiles.SetValues(llvm::make_range(
            old_body->getArguments().begin() + arg_idx,
            old_body->getArguments().begin() + arg_idx + num_vectors));
        arg_idx += num_vectors;
        RollVectorsOp rolled_op =
            assemble(builder, vty, *layout, tiles, ctx.target_shape);
        old_arg.replaceUsesWithIf(rolled_op, [&](OpOperand &operand) {
          return operand.getOwner() != rolled_op;
        });
      } else {
        TPU_ASSERT_OP(!layout.has_value());
        old_arg.replaceAllUsesWith(old_body->getArgument(arg_idx));
        ++arg_idx;
      }
    }
    old_body->eraseArguments(0, num_old_args);
    return success();
  };

  const auto before_status = tile_body_args(while_op.getBeforeBody(),
                                            new_op.getBeforeBody(), layouts_in);
  if (before_status.failed()) return before_status;
  new_op.getBefore().takeBody(while_op.getBefore());

  const auto after_status = tile_body_args(while_op.getAfterBody(),
                                           new_op.getAfterBody(), layouts_out);
  if (after_status.failed()) return after_status;
  new_op.getAfter().takeBody(while_op.getAfter());

  builder.setInsertionPointAfter(new_op);
  int64_t res_idx = 0;
  SmallVector<Value> rolled_results;
  for (auto [result, layout] :
       llvm::zip_equal(while_op.getResults(), layouts_out)) {
    if (const auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(res_idx + num_vectors, new_op.getResults().size());
      tiles.SetValues(llvm::make_range(
          new_op.getResults().begin() + res_idx,
          new_op.getResults().begin() + res_idx + num_vectors));
      res_idx += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      rolled_results.push_back(rolled_op);
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      rolled_results.push_back(new_op.getResult(res_idx));
      ++res_idx;
    }
  }

  while_op.replaceAllUsesWith(rolled_results);
  while_op.erase();
  return success();
}

LogicalResult scf_condition_rule(RewriteContext &ctx, Operation &op,
                                 const ArrayRef<Layout> layouts_in,
                                 const ArrayRef<Layout> layouts_out) {
  OpBuilder builder(&op);
  auto condition_op = cast<scf::ConditionOp>(op);
  TPU_ASSERT_EQ_OP(layouts_in.size(), condition_op.getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  SmallVector<Value> unrolled;

  for (auto [operand, layout] :
       llvm::zip_equal(condition_op.getOperands(), layouts_in)) {
    if (auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand)) {
      // When the operand has vector type, disassemble the operand.
      TPU_ASSERT_OP(layout.has_value());
      FAILUREOR_ASSIGN_OR_RETURN(
          const xla::Array<Value> tiles,
          disassemble(builder, *layout, vector_operand, ctx.target_shape));
      unrolled.append(tiles.begin(), tiles.end());
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      unrolled.push_back(operand);
    }
  }

  // Replace the old operands with unrolled operands.
  condition_op->setOperands(unrolled);
  return success();
}

LogicalResult scf_if_rule(RewriteContext &ctx, Operation &op,
                          const ArrayRef<Layout> layouts_in,
                          const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(!layouts_in.front().has_value());
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  scf::IfOp if_op = cast<scf::IfOp>(op);
  SmallVector<Layout, 4> then_yield_in_layouts;
  SmallVector<Layout, 4> else_yield_in_layouts;
  FAILUREOR_ASSIGN_OR_RETURN(
      then_yield_in_layouts,
      getInLayouts(*if_op.thenYield(), ctx.target_shape));
  if (!if_op.getElseRegion().empty()) {
    FAILUREOR_ASSIGN_OR_RETURN(
        else_yield_in_layouts,
        getInLayouts(*if_op.elseYield(), ctx.target_shape));
  }
  int out_idx = 0;
  for (auto [then_layout, else_layout, result_layout, result] :
       llvm::zip_equal(then_yield_in_layouts, else_yield_in_layouts,
                       layouts_out, op.getResults())) {
    if (auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(then_layout.has_value());
      TPU_ASSERT_OP(else_layout.has_value());
      TPU_ASSERT_OP(result_layout.has_value());
      if (result_layout.value() != then_layout.value()) {
        return op.emitOpError(
                   "Not implemented: yield layout from then branch does not "
                   "match with output layout ")
               << out_idx;
      }
      if (result_layout.value() != else_layout.value()) {
        return op.emitOpError(
                   "Not implemented: yield layout from else branch does not "
                   "match with output layout ")
               << out_idx;
      }
    } else {
      TPU_ASSERT_EQ_OP(then_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(else_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(result_layout, kNoLayout);
    }
    ++out_idx;
  }
  if (failed(applyLayoutBlock(ctx, *if_op.thenBlock()))) {
    return failure();
  }
  if (if_op.getElseRegion().empty()) {
    TPU_ASSERT_EQ_OP(if_op->getNumResults(), 0);
    TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
    return success();
  }
  if (failed(applyLayoutBlock(ctx, *if_op.elseBlock()))) {
    return failure();
  }

  // Apply layout to results after applying layout in the true and false
  // regions.
  if (if_op.getNumResults() == 0) {
    TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
    return success();
  }
  TPU_ASSERT_EQ_OP(if_op.getNumResults(), layouts_out.size());
  // If scf.if has results, it should have both non-empty true and false
  // regions.
  TPU_ASSERT_OP(!if_op.getThenRegion().empty() &&
                !if_op.getElseRegion().empty());

  // Move true and false regions to the new if op whose result has same type and
  // layout as yield operand's.
  auto new_op = builder.create<scf::IfOp>(
      TypeRange(if_op.thenYield().getResults()), if_op.getCondition(),
      /*withElseRegion =*/true);
  moveAllRegions(*if_op, *new_op);

  int64_t index = 0;
  SmallVector<Value> rolled_results;
  for (auto [result, layout] :
       llvm::zip_equal(if_op.getResults(), layouts_out)) {
    if (const auto vty = dyn_cast<VectorType>(result.getType())) {
      // When the result has a vector type, assemble the result.
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(index + num_vectors, new_op.getResults().size());
      tiles.SetValues(
          llvm::make_range(new_op.getResults().begin() + index,
                           new_op.getResults().begin() + index + num_vectors));
      index += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      rolled_results.push_back(rolled_op);
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      rolled_results.push_back(new_op.getResult(index));
      ++index;
    }
  }
  if_op.replaceAllUsesWith(rolled_results);
  if_op.erase();
  return success();
}

LogicalResult yield_rule(RewriteContext &ctx, Operation &op,
                         const ArrayRef<Layout> layouts_in,
                         const ArrayRef<Layout> layouts_out) {
  OpBuilder builder(&op);
  TPU_ASSERT_EQ_OP(layouts_in.size(), op.getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  if (op.getNumOperands() == 0) {
    return success();
  }
  SmallVector<Value> unrolled;
  for (auto [operand, layout] :
       llvm::zip_equal(op.getOperands(), layouts_in)) {
    if (auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand)) {
      // When the operand has vector type, disassemble the operand.
      TPU_ASSERT_OP(layout.has_value());
      FAILUREOR_ASSIGN_OR_RETURN(
          const xla::Array<Value> tiles,
          disassemble(builder, *layout, vector_operand, ctx.target_shape));
      unrolled.append(tiles.begin(), tiles.end());
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      unrolled.push_back(operand);
    }
  }

  // Replace the old operands with unrolled operands.
  op.setOperands(unrolled);
  return success();
}

LogicalResult tpu_load_rule(RewriteContext &ctx, Operation &op,
                            const ArrayRef<Layout> layouts_in,
                            const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(llvm::none_of(layouts_in,
                              [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_out = *layouts_out.front();
  // We expect the result is already a native-sized vreg.
  // TODO(b/300493694): Support other bitwidths
  if (layout_out.bitwidth() != 32) {
    return op.emitOpError("Not implemented: Only 32-bit loads supported");
  }
  tpu::LoadOp load_op = cast<tpu::LoadOp>(op);
  if (layout_out != VectorLayout(32, {0, 0}, ctx.target_shape,
                                 VectorLayout::ImplicitDim::kNone)) {
    return op.emitOpError("Invalid output layout for ") << load_op->getName();
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<int64_t> indices,
      getIntConstsFromOperandRange(load_op.getIndices()));
  TPU_ASSERT_EQ_OP(indices.size(), 2);
  if (indices[1] % ctx.target_shape[1] != 0) {
    return op.emitOpError("Not implemented: Lane index is not a multiple of ")
           << ctx.target_shape[1];
  }

  OpBuilder builder(op.getContext());
  builder.setInsertionPointAfter(&op);
  const RollVectorsOp roll_vectors_op =
      assemble(builder, load_op.getResult().getType(), layout_out,
               {{load_op.getResult()}}, ctx.target_shape);
  load_op->replaceUsesWithIf(roll_vectors_op, [&](OpOperand &operand) {
    return operand.getOwner() != roll_vectors_op;
  });
  return success();
}

LogicalResult strided_op_rule_impl(RewriteContext &ctx, Operation &op,
                                   Value base_ref, ValueRange indices,
                                   const VectorType &vty,
                                   const VectorLayout &layout,
                                   const ArrayRef<int32_t> &strides) {
  if (!isa<tpu::StridedLoadOp, tpu::StridedStoreOp>(op)) {
    return op.emitOpError("Not implemented: Unsupported strided op")
           << op.getName();
  }
  if (layout != VectorLayout(32, {0, 0}, ctx.target_shape,
                             VectorLayout::ImplicitDim::kNone)) {
    return op.emitOpError("Not implemented: Unsupported vector layout in ")
           << op.getName();
  }
  const auto base_ty = getMemRefType(base_ref);
  auto rank = base_ty.getRank();
  CHECK_EQ(rank, indices.size());
  CHECK_EQ(rank, strides.size());
  CHECK_EQ(rank, vty.getShape().size());
  if (rank < 2) {
    return op.emitOpError("Not implemented: Stride on 1D vector");
  }
  auto mem_layout = dyn_cast<TiledLayoutAttr>(base_ty.getLayout());
  if (!mem_layout) {
    return op.emitOpError("Expected a tiled memref");
  }
  auto tile_strides = mem_layout.getTileStrides();

  // Currently we hold constraints that the last dim size of memref needs to be
  // exactly same as the lane size of native vreg and the memref has never
  // been sliced before on the last dim. In other words, the original base
  // memref's shape needs to be (..., target_shape[1]).
  if (base_ty.getShape()[rank - 1] != ctx.target_shape[1] ||
      tile_strides.take_back(2) != ArrayRef<int64_t>{1, 1}) {
    return op.emitOpError("Not Implemented: The last dim size is not ")
           << ctx.target_shape[1] << " in original base memref";
  }
  if (strides[rank - 1] != 1) {
    return op.emitOpError("Not Implemented: Stride on last dim is not 1");
  }
  auto last_idx = getIntConst(indices[rank - 1], /*silent=*/true);
  if (failed(last_idx)) {
    return op.emitOpError("Not Implemented: Dynamic index on last dim");
  } else if (last_idx.value() != 0) {
    return op.emitOpError("Not Implemented: Index on last dim is not 0");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);

  VectorType vreg_ty =
      getNativeVregType(vty.getElementType(), ctx.target_shape);

  bool is_load_op = true;
  xla::Array<Value> tiles(
      layout.tileArrayShape(vty.getShape(), ctx.target_shape));
  if (auto store_op = dyn_cast<tpu::StridedStoreOp>(op)) {
    is_load_op = false;
    FAILUREOR_ASSIGN_OR_RETURN(
        tiles, disassemble(builder, layout, store_op.getValueToStore(),
                           ctx.target_shape));
  }

  tiles.Each([&](absl::Span<const int64_t> tile_idxs, Value *v) {
    CHECK_EQ(tile_idxs.size(), rank);
    SmallVector<Value> idxs(rank);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t stride = (i < rank - 2)
                           ? strides[i]
                           : (strides[i] * ctx.target_shape[i - rank + 2]);
      idxs[i] = builder.create<arith::AddIOp>(
          indices[i], IdxConst(tile_idxs[i] * stride, builder, op.getLoc()));
    }
    SmallVector<bool> sublane_mask(ctx.target_shape[0], true);
    int64_t sublane_rem = vty.getDimSize(rank - 2) % ctx.target_shape[0];
    if (sublane_rem > 0 && tile_idxs[rank - 2] == tiles.dim(rank - 2) - 1) {
      for (int64_t i = sublane_rem; i < ctx.target_shape[0]; ++i) {
        sublane_mask[i] = false;
      }
    }
    const auto sublane_mask_attr =
        DenseBoolArrayAttr::get(op.getContext(), sublane_mask);
    if (is_load_op) {
      *v = builder.create<tpu::LoadOp>(
          vreg_ty, base_ref, idxs, sublane_mask_attr,
          builder.getI32IntegerAttr(strides[rank - 2]));
    } else {
      builder.create<tpu::StoreOp>(
          *v, base_ref, idxs, sublane_mask_attr,
          /*mask=*/nullptr, builder.getI32IntegerAttr(strides[rank - 2]));
    }
  });
  if (is_load_op) {
    op.replaceAllUsesWith(
        assemble(builder, vty, layout, std::move(tiles), ctx.target_shape));
  }
  op.erase();
  return success();
}

// TODO(jevinjiang): maybe unify with vector load?
LogicalResult tpu_strided_load_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_OP(llvm::none_of(layouts_in,
                              [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_out = *layouts_out.front();
  auto load_op = cast<tpu::StridedLoadOp>(op);
  const auto vty = cast<VectorType>(load_op.getResult().getType());
  return strided_op_rule_impl(ctx, op, load_op.getBase(), load_op.getIndices(),
                              vty, layout_out, load_op.getStrides());
}

// TODO(jevinjiang): maybe unify with vector store?
LogicalResult tpu_strided_store_rule(RewriteContext &ctx, Operation &op,
                                     const ArrayRef<Layout> layouts_in,
                                     const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(llvm::none_of(layouts_in.drop_front(),
                              [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);

  const VectorLayout &to_store_layout = *layouts_in.front();
  auto store_op = cast<tpu::StridedStoreOp>(op);
  const auto vty = store_op.getValueToStore().getType();
  return strided_op_rule_impl(ctx, op, store_op.getBase(),
                              store_op.getIndices(), vty, to_store_layout,
                              store_op.getStrides());
}

LogicalResult tpu_matmul_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 3);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(
      llvm::all_of(layouts_in, [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto matmul_op = cast<tpu::MatmulOp>(op);
  if (matmul_op.getTransposeRhs()) {
    return op.emitOpError(
        "Transposition must have been erased into dimension numbers during "
        "canonicalization");
  }

  auto dimension_numbers = matmul_op.getDimensionNumbers();
  if (!dimension_numbers.has_value()) {
    return op.emitOpError(
        "Dimension numbers must be provided, ensure canonicalization has been "
        "run.");
  }
  auto transposed_mkn = isTransposedMatmul(dimension_numbers.value());
  if (!transposed_mkn.has_value()) {
    return op.emitOpError(
        "Dimension numbers must be MKN, ensure canonicalization has been "
        "run.");
  }
  auto [transpose_lhs, transpose_rhs] = transposed_mkn.value();
  if (transpose_lhs) {
    return op.emitOpError(
        "Transposition of LHS is not supported in apply_vector_layout, ensure "
        "canonicalization has been run.");
  }

  auto &layout_lhs = *layouts_in[0];
  auto &layout_rhs = *layouts_in[1];
  auto &layout_acc = *layouts_in[2];
  auto &layout_out = *layouts_out[0];

  const std::array<std::reference_wrapper<const VectorLayout>, 4> all_layouts =
      {layout_lhs, layout_rhs, layout_acc, layout_out};
  for (const VectorLayout &layout : all_layouts) {
    for (const LayoutOffset offset : layout.offsets()) {
      if (offset.value_or(0) != 0) {
        return op.emitOpError("Not implemented: Unaligned layout in matmul");
      }
    }
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  TypedValue<VectorType> lhs, rhs, acc, res;
  if (auto tpu_matmul_op = dyn_cast<tpu::MatmulOp>(op)) {
    lhs = tpu_matmul_op.getLhs();
    rhs = tpu_matmul_op.getRhs();
    acc = tpu_matmul_op.getAcc();
    res = tpu_matmul_op.getResult();
  } else {
    return op.emitOpError("Expected a tpu::MatmulOp");
  }

  for (const std::optional<VectorLayout> &layout_opt : layouts_in) {
    auto layout = *layout_opt;
    if (layout.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
      return op.emitOpError(
          "Not implemented: Unsupported matmul operand layout");
    }
    if (!layout.hasNativeTiling(ctx.target_shape)) {
      return op.emitOpError(
          "Not implemented: Unsupported matmul operand tiling");
    }
  }
  if (acc.getType().getElementType().getIntOrFloatBitWidth() != 32) {
    return op.emitOpError("Not implemented: Non-32-bit matmul acc");
  }
  const ArrayRef<int64_t> lhs_shape = lhs.getType().getShape();
  const ArrayRef<int64_t> rhs_shape = rhs.getType().getShape();
  // TODO(tlongeri): This should be part of the tpu::MatmulOp verifier
  TPU_ASSERT_EQ_OP(lhs_shape.size(), 2);
  TPU_ASSERT_EQ_OP(rhs_shape.size(), 2);

  const int64_t padded_lhs_rows =
      llvm::alignTo(lhs_shape[0], layout_lhs.tiling()[0]);
  const int64_t padded_lhs_cols =
      llvm::alignTo(lhs_shape[1], layout_lhs.tiling()[1]);
  const int64_t padded_rhs_rows =
      llvm::alignTo(rhs_shape[0], layout_rhs.tiling()[0]);
  const int64_t padded_rhs_cols =
      llvm::alignTo(rhs_shape[1], layout_rhs.tiling()[1]);

  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> lhs_vregs,
      disassemble(builder, layout_lhs, lhs, ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> acc_vregs,
      disassemble(builder, layout_acc, acc, ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> rhs_vregs,
      disassemble(builder, layout_rhs, rhs, ctx.target_shape));
  TPU_ASSERT_EQ_OP(padded_lhs_rows, lhs_vregs.dim(0) * layout_lhs.tiling()[0]);
  TPU_ASSERT_EQ_OP(padded_rhs_rows, rhs_vregs.dim(0) * layout_rhs.tiling()[0]);

  auto lhs_zeros_vreg =
      getZerosVector(builder, cast<VectorType>(lhs_vregs.begin()->getType()));
  auto rhs_zeros_vreg =
      getZerosVector(builder, cast<VectorType>(rhs_vregs.begin()->getType()));
  auto acc_zeros_vreg =
      getZerosVector(builder, cast<VectorType>(acc_vregs.begin()->getType()));

  // Only mask out the paddings on contracting dim of LHS and RHS.
  RETURN_IF_FAILED(
      maskNativeTilingVregs(builder, lhs_vregs, ctx.target_shape,
                            /*padding_bottom=*/0,
                            /*padding_right=*/padded_lhs_cols - lhs_shape[1]));
  if (transpose_rhs) {
    RETURN_IF_FAILED(maskNativeTilingVregs(
        builder, rhs_vregs, ctx.target_shape,
        /*padding_bottom=*/0,
        /*padding_right=*/padded_rhs_cols - rhs_shape[1]));
  } else {
    RETURN_IF_FAILED(
        maskNativeTilingVregs(builder, rhs_vregs, ctx.target_shape,
                              /*padding_bottom=*/padded_rhs_rows - rhs_shape[0],
                              /*padding_right=*/0));
  }

  // At this point, all paddings on vregs are masked out. For now, we
  // append zero vregs to make LHS's second dim, both RHS's dims and ACC's
  // second dim to be a multiple of mxu_size.
  auto mxu_contracting_size = ctx.mxu_shape[0];
  auto mxu_noncontracting_size = ctx.mxu_shape[1];
  if (lhs.getType().getElementType().isSignlessInteger(4) &&
      rhs.getType().getElementType().isSignlessInteger(4)) {
    mxu_contracting_size *= 2;
  }
  auto rhs_row_size = mxu_contracting_size;
  auto rhs_col_size = mxu_noncontracting_size;
  if (transpose_rhs) {
    rhs_row_size = mxu_noncontracting_size;
    rhs_col_size = mxu_contracting_size;
  }
  CHECK_EQ(rhs_row_size % ctx.target_shape[1], 0);
  CHECK_EQ(rhs_col_size % ctx.target_shape[1], 0);

  // Here, a single group corresponds to a single matmul invocation in unrolled
  // code. The RHS group matches the MXU shape.
  auto lhs_col_vregs_per_group = mxu_contracting_size / ctx.target_shape[1];
  auto rhs_row_vregs_per_group =
      rhs_row_size / (ctx.target_shape[0] * layout_rhs.packing());
  auto rhs_col_vregs_per_group = rhs_col_size / ctx.target_shape[1];
  auto acc_col_vregs_per_group = mxu_noncontracting_size / ctx.target_shape[1];
  int64_t target_lhs_col_vregs =
      llvm::alignTo(lhs_vregs.dim(1), lhs_col_vregs_per_group);
  int64_t target_rhs_row_vregs =
      llvm::alignTo(rhs_vregs.dim(0), rhs_row_vregs_per_group);
  int64_t target_rhs_col_vregs =
      llvm::alignTo(rhs_vregs.dim(1), rhs_col_vregs_per_group);
  int64_t target_acc_col_vregs =
      llvm::alignTo(acc_vregs.dim(1), acc_col_vregs_per_group);

  xla::Array<Value> target_lhs_vregs({lhs_vregs.dim(0), target_lhs_col_vregs},
                                     lhs_zeros_vreg);
  xla::Array<Value> target_rhs_vregs(
      {target_rhs_row_vregs, target_rhs_col_vregs}, rhs_zeros_vreg);
  xla::Array<Value> target_acc_vregs(
      {lhs_vregs.dim(0) * layout_lhs.packing(), target_acc_col_vregs},
      acc_zeros_vreg);
  target_lhs_vregs.UpdateSlice(lhs_vregs, {0, 0});
  target_rhs_vregs.UpdateSlice(rhs_vregs, {0, 0});
  target_acc_vregs.UpdateSlice(acc_vregs, {0, 0});

  // Now we can regroup vregs from target vregs.
  const auto lhs_col_ty = VectorType::get(
      {padded_lhs_rows, mxu_contracting_size}, lhs.getType().getElementType());
  const auto acc_col_ty =
      VectorType::get({padded_lhs_rows, mxu_noncontracting_size},
                      acc.getType().getElementType());
  const ArrayAttr lhs_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_lhs)});
  const ArrayAttr rhs_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_rhs)});
  const ArrayAttr acc_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_acc)});

  int64_t nk = llvm::divideCeil(lhs_shape[1], mxu_contracting_size);
  CHECK_EQ(nk, target_lhs_vregs.dim(1) / lhs_col_vregs_per_group);
  SmallVector<tpu::RollVectorsOp> lhs_cols(nk);
  for (int64_t i = 0; i < nk; ++i) {
    const xla::Array<Value> col_vregs = target_lhs_vregs.Slice(
        {0, i * lhs_col_vregs_per_group},
        {target_lhs_vregs.dim(0), (i + 1) * lhs_col_vregs_per_group});
    lhs_cols[i] = builder.create<tpu::RollVectorsOp>(
        op.getLoc(), lhs_col_ty, XlaArrayToFlatArrayRef(col_vregs));
    lhs_cols[i]->setAttr("out_layout", lhs_layout_attr);
  }
  const auto rhs_group_ty = VectorType::get({rhs_row_size, rhs_col_size},
                                            rhs.getType().getElementType());
  const int64_t rhs_vregs_per_group =
      rhs_row_vregs_per_group * rhs_col_vregs_per_group;

  int64_t nj;
  if (transpose_rhs) {
    nj = llvm::divideCeil(rhs_shape[0], rhs_row_size);
    CHECK_EQ(nk, llvm::divideCeil(rhs_shape[1], rhs_col_size));
    CHECK_EQ(nk, target_rhs_vregs.dim(1) / rhs_col_vregs_per_group);
    target_rhs_vregs.Reshape({nj, rhs_vregs_per_group / rhs_col_vregs_per_group,
                              nk, rhs_col_vregs_per_group});
    target_rhs_vregs.TransposeDimensions({2, 0, 1, 3});
    target_rhs_vregs.Reshape({nk, nj, rhs_vregs_per_group});
  } else {
    nj = llvm::divideCeil(rhs_shape[1], rhs_col_size);
    CHECK_EQ(nk, llvm::divideCeil(rhs_shape[0], rhs_row_size));
    CHECK_EQ(nk, target_rhs_vregs.dim(0) / rhs_row_vregs_per_group);
    target_rhs_vregs.Reshape({nk, rhs_vregs_per_group / rhs_col_vregs_per_group,
                              nj, rhs_col_vregs_per_group});
    target_rhs_vregs.TransposeDimensions({0, 2, 1, 3});
    target_rhs_vregs.Reshape({nk, nj, rhs_vregs_per_group});
  }

  const tpu::ContractPrecisionAttr precision_attr =  // May be null
      op.getAttrOfType<tpu::ContractPrecisionAttr>("precision");
  const tpu::DotDimensionNumbersAttr dot_dimension_numbers_attr =
      defaultDimensionNumbers(builder, false, transpose_rhs);
  for (int64_t j = 0; j < nj; ++j) {
    for (int64_t k = 0; k < nk; ++k) {
      // TODO(tlongeri): there should be a way to slice without copying
      xla::Array<Value> rhs_group = target_rhs_vregs.Slice(
          {k, j, 0}, {k + 1, j + 1, rhs_vregs_per_group});
      auto rhs_rolled_group = builder.create<tpu::RollVectorsOp>(
          op.getLoc(), rhs_group_ty, XlaArrayToFlatArrayRef(rhs_group));
      rhs_rolled_group->setAttr("out_layout", rhs_layout_attr);
      const xla::Array<Value> acc_col_vregs = target_acc_vregs.Slice(
          {0, j * acc_col_vregs_per_group},
          {target_acc_vregs.dim(0), (j + 1) * acc_col_vregs_per_group});
      auto acc_col = builder.create<tpu::RollVectorsOp>(
          op.getLoc(), acc_col_ty, XlaArrayToFlatArrayRef(acc_col_vregs));
      acc_col->setAttr("out_layout", acc_layout_attr);
      auto new_acc_col = builder.create<tpu::MatmulOp>(
          op.getLoc(), acc_col_ty, lhs_cols[k], rhs_rolled_group, acc_col,
          /*transpose_lhs=*/false, /*transpose_rhs=*/false, precision_attr,
          dot_dimension_numbers_attr);
      auto new_acc_vregs = builder.create<tpu::UnrollVectorsOp>(
          op.getLoc(),
          TypeRange(ValueRange(XlaArrayToFlatArrayRef(acc_col_vregs))),
          new_acc_col);
      new_acc_vregs->setAttr("in_layout", acc_layout_attr);
      updateSliceFromRange(
          target_acc_vregs, new_acc_vregs->getResults(),
          {0, j * acc_col_vregs_per_group},
          {target_acc_vregs.dim(0), (j + 1) * acc_col_vregs_per_group});
    }
  }
  op.replaceAllUsesWith(
      assemble(
          builder, res.getType(), layout_out,
          target_acc_vregs.Slice({0, 0}, {acc_vregs.dim(0), acc_vregs.dim(1)}),
          ctx.target_shape)
          .getOperation());
  op.erase();
  return success();
}

LogicalResult tpu_store_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  TPU_ASSERT_OP(layouts_in.front().has_value());  // value to store layout
  TPU_ASSERT_OP(llvm::none_of(layouts_in.drop_front(),
                              [&](const Layout &l) { return l.has_value(); }));
  OpBuilder builder(&op);
  const VectorLayout &to_store_layout = *layouts_in.front();
  // We expect the value to store is already a native-sized vreg.
  if (to_store_layout.bitwidth() != 32) {
    return op.emitOpError("Not implemented: Only 32-bit loads supported");
  }
  TPU_ASSERT_OP(to_store_layout ==
                VectorLayout(32, {0, 0}, ctx.target_shape,
                             VectorLayout::ImplicitDim::kNone));
  tpu::StoreOp store_op = cast<tpu::StoreOp>(op);
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<int64_t> indices,
      getIntConstsFromOperandRange(store_op.getIndices()));
  TPU_ASSERT_EQ_OP(indices.size(), 2);
  if (indices[1] % ctx.target_shape[1] != 0) {
    return op.emitOpError("Not implemented: Lane index is not a multiple of ")
           << ctx.target_shape[1];
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> tiles,
      disassemble(builder, to_store_layout, store_op.getValueToStore(),
                  ctx.target_shape));
  TPU_ASSERT_OP((tiles.dimensions() == xla::DimensionVector{1, 1}));
  store_op.getValueToStoreMutable().assign(tiles({0, 0}));
  return success();
}

LogicalResult tpu_bitcast_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &in_layout = *layouts_in.front();
  const VectorLayout &out_layout = *layouts_out.front();
  auto in_bitwidth = in_layout.bitwidth();
  auto out_bitwidth = out_layout.bitwidth();
  auto in_tiling = in_layout.tiling();
  auto out_tiling = out_layout.tiling();
  in_tiling[0] *= in_bitwidth;
  out_tiling[0] *= out_bitwidth;
  if (in_tiling != out_tiling) {
    return op.emitOpError(
        "Expected tilings are the same after multiplying the "
          "second-minor dimension by the ratio of bitwidths.");
  }
  auto in_offsets = in_layout.offsets();
  auto out_offsets = out_layout.offsets();
  if (!out_offsets[0].has_value() && in_bitwidth > out_bitwidth) {
    return op.emitOpError(
        "Expected no replicated offset on 2nd minor dimension of output when "
        "bitwidth is decreased.");
  }
  if (in_offsets[0].has_value() != out_offsets[0].has_value() ||
      in_offsets[0].value_or(0) * in_bitwidth !=
          out_offsets[0].value_or(0) * out_bitwidth ||
      in_offsets[1] != out_offsets[1]) {
    return op.emitOpError(
        "Expected offsets are the same after multiplying the "
          "second-minor dimension by the ratio of bitwidths.");
  }
  if (in_layout.implicit_dim() != out_layout.implicit_dim()) {
    return op.emitOpError(
        "Expected same implicit dim for input and output layout");
  }
  auto bitcast_op = cast<tpu::BitcastOp>(op);
  const auto out_ty = bitcast_op.getResult().getType();
  if (in_bitwidth != out_bitwidth) {
    if (in_layout.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
      return op.emitOpError("Expected no implicit dim when bitwidth changes");
    }
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  const auto native_vreg_ty =
      getNativeVregType(out_ty.getElementType(), ctx.target_shape);
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> in_tiles,
      disassemble(builder, in_layout, bitcast_op.getInput(), ctx.target_shape));
  xla::Array<Value> out_tiles(in_tiles.dimensions());
  out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
    const Value in_tile = in_tiles(idxs);
    *v = builder.create<tpu::BitcastVregOp>(native_vreg_ty, in_tile);
  });
  bitcast_op.replaceAllUsesWith(
      assemble(builder, out_ty, out_layout, out_tiles, ctx.target_shape)
          .getOperation());
  bitcast_op.erase();
  return success();
}

LogicalResult tpu_trace_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
    return op.emitOpError(
        "Not implemented: tpu.traced_block with inputs or outputs");
  }
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  // We don't modify the op, but we do rewrite the branch bodies.
  TPU_ASSERT_EQ_OP(op.getNumRegions(), 1);
  Region &region = op.getRegion(0);
  TPU_ASSERT_OP(region.hasOneBlock());
  Block &block = region.front();
  return applyLayoutBlock(ctx, block);
}

LogicalResult tpu_assume_layout_rule(RewriteContext &ctx, Operation &op,
                                     const ArrayRef<Layout> layouts_in,
                                     const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(op.getNumOperands(), 1);
  TPU_ASSERT_EQ_OP(op.getNumResults(), 1);
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  if (layouts_in[0] != layouts_out[0]) {
    return op.emitOpError("Expected same input and output layout");
  }
  OpBuilder builder(&op);
  auto val = op.getOperand(0);
  auto layout = layouts_in[0];
  const auto vty = cast<VectorType>(val.getType());
  SmallVector<int64_t> layout_shape =
      layout->tileArrayShape(vty.getShape(), ctx.target_shape);
  const int64_t num_vectors = ShapedType::getNumElements(layout_shape);
  VectorType vreg_ty =
      getNativeVregType(vty.getElementType(), ctx.target_shape);
  // We can not use disassemble here because the val is block argument.
  auto unrolled_op = builder.create<tpu::UnrollVectorsOp>(
      val.getLoc(), SmallVector<Type>(num_vectors, vreg_ty), val);

  op.replaceAllUsesWith(assemble(builder, vty, *layout,
                                 XlaArrayFromShapeAndValues<Value>(
                                     layout_shape, unrolled_op->getResults()),
                                 ctx.target_shape));
  op.erase();
  return success();
}

// TODO(b/347016737): Deprecate tpu.rotate and only use tpu.dynamic_rotate. So
// we do not need template for the op type and to explicitly force amount
// argument to dynamic.
template <typename OpTy>
LogicalResult rotate_rule_impl(RewriteContext &ctx, OpTy op, Value amount,
                               const VectorLayout &layout_in,
                               const VectorLayout &layout_out) {
  auto layout = VectorLayout(32, {0, 0}, ctx.target_shape,
                             VectorLayout::ImplicitDim::kNone);
  if (layout_in != layout) {
    return op.emitOpError("Not implemented: unsupported layout for input");
  }
  if (layout_out != layout) {
    return op.emitOpError("Not implemented: unsupported layout for output");
  }
  auto vty = op.getResult().getType();
  if (vty.getRank() < 2) {
    return op.emitOpError("Not implemented: unsupported 1D shape");
  }
  if (*(vty.getShape().end() - 2) % *(layout.tiling().end() - 2) != 0 ||
      *(vty.getShape().end() - 1) % *(layout.tiling().end() - 1) != 0) {
    return op.emitOpError("Not implemented: unsupported unaliged shape");
  }

  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());

  VectorType res_vreg_ty =
      getNativeVregType(vty.getElementType(), ctx.target_shape);
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> in_tiles,
      disassemble(builder, layout_in, op.getValue(), ctx.target_shape));

  const VectorType i32_vreg =
      getNativeVregType(builder.getI32Type(), ctx.target_shape);

  // Some helper functions for math ops.
  auto mlirI32Const = [&](int d) {
    return builder.create<arith::ConstantOp>(
        builder.getIntegerAttr(builder.getI32Type(), d));
  };
  auto mlirIndexConst = [&](int d) {
    return builder.create<arith::ConstantOp>(
        builder.getIntegerAttr(builder.getIndexType(), d));
  };
  auto modI = [&](const Value &v, unsigned d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return mlirI32Const(cst.value() % d);
    }
    return builder.create<arith::RemUIOp>(v, mlirI32Const(d));
  };
  auto divI = [&](const Value &v, unsigned d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return mlirI32Const(cst.value() / d);
    }
    return builder.create<arith::DivUIOp>(v, mlirI32Const(d));
  };
  auto addI = [&](const Value &v, unsigned d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return mlirI32Const(cst.value() + d);
    }
    return builder.create<arith::AddIOp>(v, mlirI32Const(d));
  };

  // A helper function that creates a VMASK with false flags to bottom (dim = 0)
  // or right (dim = 1) where the flag count corresponds to the (dim_size -
  // padding). If stride is provided, the padding value is sequentially
  // increased by the stride value along the dim.
  //
  // For example, assume VMASK shape is (4, 8)
  //
  // getVmaskByPaddingEnd(padding=3, dim=1) creates:
  //  [T, T, T, T, T, F, F, F]
  //  [T, T, T, T, T, F, F, F]
  //  [T, T, T, T, T, F, F, F]
  //  [T, T, T, T, T, F, F, F]
  //
  // getVmaskByPaddingEnd(padding=3, dim=1, stride=1) creates:
  //  [T, T, T, T, T, F, F, F]
  //  [T, T, T, T, T, T, F, F]
  //  [T, T, T, T, T, T, T, F]
  //  [T, T, T, T, T, T, T, T]
  auto getVmaskByPaddingEnd = [&](Value padding, int dim, int stride = 0) {
    CHECK(dim == 0 || dim == 1);
    Value padding_vreg;
    if (auto padding_cst = getIntConst(padding, /*silent=*/true);
        succeeded(padding_cst)) {
      CHECK_GE(padding_cst.value(), 0);
      CHECK_LE(padding_cst.value(), ctx.target_shape[dim]);
      padding_vreg = builder.create<arith::ConstantOp>(DenseElementsAttr::get(
          i32_vreg, builder.getI32IntegerAttr(padding_cst.value())));
    } else {
      padding_vreg = builder.create<vector::BroadcastOp>(i32_vreg, padding);
    }

    if (stride > 0) {
      auto offset = builder.create<arith::MulIOp>(
          i32_vreg,
          builder.create<tpu::IotaOp>(
              i32_vreg, builder.getI32IntegerAttr(dim == 0 ? 1 : 0)),
          builder.create<arith::ConstantOp>(DenseElementsAttr::get(
              i32_vreg, builder.getI32IntegerAttr(stride))));
      padding_vreg =
          builder.create<arith::AddIOp>(i32_vreg, padding_vreg, offset);
    }
    return builder.create<arith::CmpIOp>(
        arith::CmpIPredicate::slt,
        builder.create<tpu::IotaOp>(i32_vreg, builder.getI32IntegerAttr(dim)),
        padding_vreg);
  };

  // Apply rotation on each vreg with the assumption that shift <= VREG dim size
  // and blend the data from contiguous vregs to emulate circular rotation.
  auto rotateOnTilingDim = [&](const xla::Array<Value> &vregs,
                               const Value &shift, int axis, int stride = 0) {
    if (auto shift_cst = getIntConst(shift, /*silent=*/true);
        succeeded(shift_cst)) {
      if (shift_cst.value() == 0 && stride == 0) {
        return vregs;
      }
    }
    int tiling_dim = axis - (vregs.num_dimensions() - 2);
    CHECK((tiling_dim == 0 && stride == 0) || (tiling_dim == 1 && stride >= 0));
    xla::Array<Value> result(vregs.dimensions());
    auto chunks = split(vregs, axis);
    for (int64_t i = 0; i < chunks.size(); ++i) {
      chunks[i].Each([&](absl::Span<const int64_t> idxs, Value *v) {
        auto stride_attr =
            stride > 0 ? builder.getSI32IntegerAttr(stride) : nullptr;
        auto stride_dimension_attr =
            stride > 0 ? builder.getSI32IntegerAttr(0) : nullptr;
        *v = builder.create<tpu::DynamicRotateOp>(res_vreg_ty, *v, shift,
                                                  tiling_dim, stride_attr,
                                                  stride_dimension_attr);
      });
    }
    auto mask = getVmaskByPaddingEnd(shift, tiling_dim, stride);
    xla::Array<Value> last_chunk_copy(chunks[chunks.size() - 1]);
    for (int64_t i = chunks.size() - 1; i > 0; --i) {
      chunks[i].Each([&](absl::Span<const int64_t> idxs, Value *v) {
        *v = builder.create<arith::SelectOp>(mask, chunks[i - 1](idxs), *v);
      });
    }
    chunks[0].Each([&](absl::Span<const int64_t> idxs, Value *v) {
      *v = builder.create<arith::SelectOp>(mask, last_chunk_copy(idxs), *v);
    });
    return concatenate(chunks, axis);
  };

  std::function<xla::Array<Value>(const xla::Array<Value> &, Value, int, int)>
      rotate;
  rotate = [&](const xla::Array<Value> &vregs, Value shift, int axis,
               int stride) {
    xla::Array<Value> result(vregs.dimensions());
    CHECK(axis >= 0 && axis < vregs.num_dimensions());
    int tiling_dim = axis - (vregs.num_dimensions() - 2);
    CHECK((tiling_dim != 1 && stride == 0) || (tiling_dim == 1 && stride >= 0));
    SmallVector<xla::Array<Value>, 4> chunks;
    // Handle rotation with static shift.
    if (auto shift_cst = getIntConst(shift, /*silent=*/true);
        succeeded(shift_cst)) {
      int64_t static_shift = shift_cst.value();
      if (tiling_dim >= 0) {
        shift = mlirI32Const(static_shift % ctx.target_shape[tiling_dim]);
        static_shift /= ctx.target_shape[tiling_dim];
        chunks = split(rotateOnTilingDim(vregs, shift, axis, stride), axis);
      } else {
        chunks = split(vregs, axis);
      }
      // Now we only need to shuffle vregs.
      for (int64_t i = 0; i < chunks.size(); ++i) {
        SmallVector<int64_t> starts(result.num_dimensions(), 0);
        starts[axis] = (i + static_shift) % result.dim(axis);
        result.UpdateSlice(chunks[i], starts);
      }
      return result;
    }
    // Handle rotation with dynamic shift.
    // TODO(jevinjiang): consider optimize with assume_multiple op.
    Value in_vreg_shift = tiling_dim >= 0
                              ? modI(shift, ctx.target_shape[tiling_dim])
                              : mlirI32Const(0);
    Value vreg_shift =
        tiling_dim >= 0 ? divI(shift, ctx.target_shape[tiling_dim]) : shift;
    result = tiling_dim >= 0
                 ? rotateOnTilingDim(vregs, in_vreg_shift, axis, stride)
                 : vregs;
    int bound = vregs.dim(axis);
    if (bound <= ctx.max_sublanes_in_scratch / ctx.target_shape[0] &&
        bound >= kMinBoundToRotateWithScratch) {
      // Use static store + dynamic load to implement dynamic shift.
      if (auto scratch_ref = getInternalScratch(
              ctx, builder, op.getLoc(),
              {ctx.max_sublanes_in_scratch / ctx.target_shape[0],
               ctx.target_shape[0], ctx.target_shape[1]},
              vty.getElementType());
          succeeded(scratch_ref)) {
        auto cst_0 = mlirIndexConst(0);
        SmallVector<Value, 3> scratch_indices(3, cst_0);
        SmallVector<bool> sublane_mask(ctx.target_shape[0], true);
        const auto sublane_mask_attr =
            DenseBoolArrayAttr::get(op.getContext(), sublane_mask);
        chunks = split(result, axis);
        chunks[0].Each([&](absl::Span<const int64_t> idxs, Value *v) {
          // Static store vregs.
          for (int i = 0; i < bound; ++i) {
            scratch_indices[0] = mlirIndexConst(i);
            builder.create<tpu::StoreOp>(chunks[i](idxs), scratch_ref.value(),
                                         scratch_indices, sublane_mask_attr,
                                         /*mask=*/nullptr,
                                         /*sublane_stride=*/nullptr);
          }
          // Dynamic load vregs back from a circular buffer.
          for (int i = 0; i < bound; ++i) {
            scratch_indices[0] = builder.create<arith::IndexCastOp>(
                builder.getIndexType(),
                modI(builder.create<arith::SubIOp>(mlirI32Const(bound + i),
                                                   vreg_shift),
                     bound));
            chunks[i](idxs) =
                builder.create<tpu::LoadOp>(v->getType(), scratch_ref.value(),
                                            scratch_indices, sublane_mask_attr,
                                            /*sublane_stride=*/nullptr);
          }
        });
        return concatenate(chunks, axis);
      }
    }
    // Convert dynamic shift to log(bound) static ops.
    int roll_by = 1;
    while (roll_by < bound) {
      auto new_result = rotate(
          result,
          mlirI32Const(tiling_dim >= 0 ? roll_by * ctx.target_shape[tiling_dim]
                                       : roll_by),
          axis, /*stride=*/0);
      auto mask = builder.create<arith::CmpIOp>(
          arith::CmpIPredicate::ne,
          builder.create<vector::BroadcastOp>(
              i32_vreg,
              builder.create<arith::AndIOp>(vreg_shift, mlirI32Const(roll_by))),
          builder.create<arith::ConstantOp>(
              DenseElementsAttr::get(i32_vreg, builder.getI32IntegerAttr(0))));
      result.Each([&](absl::Span<const int64_t> idxs, Value *v) {
        *v = builder.create<arith::SelectOp>(mask, new_result(idxs), *v);
      });
      roll_by *= 2;
    }
    return result;
  };

  xla::Array<Value> out_tiles(in_tiles.dimensions());
  const auto dim = op.getDimension();
  amount = modI(amount, vty.getDimSize(dim));

  if (op.getStride().has_value() && op.getStrideDimension().has_value()) {
    auto stride_dim = op.getStrideDimension().value();
    auto stride = op.getStride().value() % vty.getDimSize(stride_dim);
    if (stride_dim == dim) {
      return op.emitOpError(
          "Expected rotation dimension and stride dimension are not equal");
    }
    if (stride_dim == vty.getRank() - 1) {
      return op.emitOpError(
          "Not implemented: stride dimension is the minor most");
    } else if (stride_dim == vty.getRank() - 2) {
      if (dim != vty.getRank() - 1 || ctx.hardware_generation < 5) {
        return op.emitOpError(
            "Not implemented: only supported in TPU v5+ and rotation dimension "
            "is the minor most when stride dimension is the second minor most");
      }
      CHECK_GE(stride, 0);
      auto chunks = split(in_tiles, stride_dim);
      for (int64_t i = 0; i < chunks.size(); ++i) {
        Value base_amount = modI(addI(amount, ctx.target_shape[0] * i * stride),
                                 vty.getDimSize(dim));
        // After applying stride, we expect all shifts in a vreg are less or
        // equal to the vreg's lane count for now.
        if (auto base_amount_cst = getIntConst(base_amount, /*silent=*/true);
            succeeded(base_amount_cst)) {
          int64_t static_base_amount = base_amount_cst.value();
          auto max_shift_in_vreg = static_base_amount % ctx.target_shape[1] +
                                   (ctx.target_shape[0] - 1) * stride;
          if (max_shift_in_vreg > ctx.target_shape[1]) {
            return op.emitOpError("Not implemented: the max shift in a vreg ")
                   << max_shift_in_vreg << " is larger than the vreg's width "
                   << ctx.target_shape[1];
          }
        }
        SmallVector<int64_t> starts(out_tiles.num_dimensions(), 0);
        starts[stride_dim] = i;
        out_tiles.UpdateSlice(rotate(chunks[i], base_amount, dim, stride),
                              starts);
      }
    } else {
      // Split vregs along the stride dimension.
      auto chunks = split(in_tiles, stride_dim);
      for (int64_t i = 0; i < chunks.size(); ++i) {
        SmallVector<int64_t> starts(out_tiles.num_dimensions(), 0);
        starts[stride_dim] = i;
        out_tiles.UpdateSlice(
            rotate(chunks[i], addI(amount, i * stride), dim, /*stride=*/0),
            starts);
      }
    }
  } else {  // No stride.
    out_tiles = rotate(in_tiles, amount, dim, /*stride=*/0);
  }

  const RollVectorsOp rolled_op =
      assemble(builder, op.getResult().getType(), layout_out, out_tiles,
               ctx.target_shape);
  op.replaceAllUsesWith(rolled_op);
  op.erase();
  return success();
}

// TODO(b/347016737): deprecate the static rotate.
LogicalResult tpu_rotate_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null input layout");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  auto rotate_op = cast<tpu::RotateOp>(op);
  if (rotate_op.getAmount() < 0) {
    return op.emitOpError("Not implemented: shifting by negative amount");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  Value shift = builder.create<arith::ConstantOp>(
      builder.getIntegerAttr(builder.getI32Type(), rotate_op.getAmount()));
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  return rotate_rule_impl(ctx, rotate_op, shift, layout_in, layout_out);
}

LogicalResult tpu_dynamic_rotate_rule(RewriteContext &ctx, Operation &op,
                                      const ArrayRef<Layout> layouts_in,
                                      const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 2);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null layout for the value to rotate");
  }
  if (layouts_in[1].has_value()) {
    return op.emitOpError("Expected null layout for the shift");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  auto rotate_op = cast<tpu::DynamicRotateOp>(op);
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  return rotate_rule_impl(ctx, rotate_op, rotate_op.getAmount(), layout_in,
                          layout_out);
}

LogicalResult tpu_concatenate_rule(RewriteContext &ctx, Operation &op,
                                   const ArrayRef<Layout> layouts_in,
                                   const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), op.getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(
      llvm::all_of(layouts_in, [](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  OpBuilder builder(&op);
  auto concatenate_op = cast<tpu::ConcatenateOp>(op);
  const VectorType res_ty = concatenate_op.getResult().getType();
  uint32_t dimension = concatenate_op.getDimension();
  SmallVector<xla::Array<Value>> operand_vregs;
  operand_vregs.reserve(op.getNumOperands());

  std::optional<int64_t> tiling_dim;
  auto res_layout = layouts_out.front();

  TPU_ASSERT_OP(res_layout.has_value());
  auto num_untiled_dims = res_ty.getRank() - res_layout->layout_rank();

  if (res_ty.getRank() == 1 &&
      res_layout->implicit_dim() == VectorLayout::ImplicitDim::kSecondMinor) {
    tiling_dim = 1;
  } else if (dimension >= num_untiled_dims) {
    tiling_dim = dimension - num_untiled_dims;
  }

  // Op level invariants on layouts, other op level invariants are checked in
  // the verifier.
  auto res_tiling = res_layout->tiling();
  for (int i = 0; i < op.getNumOperands(); ++i) {
    auto operand = op.getOperand(i);
    if (!layouts_in[i].has_value()) {
      return op.emitOpError("Not implemented: Expected input layout");
    }
    auto const &layout = *layouts_in[i];

    if (layout.tiling() != res_tiling) {
      return op.emitOpError("Not implemented: result/input Tiling mismatch");
    }

    if (layout.implicit_dim() != res_layout->implicit_dim()) {
      return op.emitOpError("Not implemented: result/input offsets mismatch.");
    }

    if (layout.implicit_dim() != res_layout->implicit_dim()) {
      return op.emitOpError(
          "Not implemented: result/input implicit dim mismatch.");
    }

    if (i > 1) {
      auto curr_offsets = layout.offsets();
      auto last_operand_offsets = layouts_in[i - 1]->offsets();
      if (tiling_dim.has_value()) {
        // Zero out the offset in the tiling dimension for verification.
        curr_offsets[tiling_dim.value()] = 0;
        last_operand_offsets[tiling_dim.value()] = 0;
      }
      if (curr_offsets != last_operand_offsets) {
        op.emitOpError("Not implemented: non-concat dim offset mismatch.");
      }
    }

    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> vreg_array,
        disassemble(builder, layout, cast<TypedValue<VectorType>>(operand),
                    ctx.target_shape));
    operand_vregs.push_back(std::move(vreg_array));
  }

  CHECK_EQ(operand_vregs.size(), op.getNumOperands());
  SmallVector<int64_t> vreg_array_shape =
      res_layout->tileArrayShape(res_ty.getShape(), ctx.target_shape);

  // Fill out out_vregs with nulls, to avoid a problem with where we have to
  // blend with a vreg that has not been written to yet.
  xla::Array<Value> out_vregs(vreg_array_shape, nullptr);

  auto boundIdxConst =
      std::bind(IdxConst, std::placeholders::_1, builder, op.getLoc());

  // Handle the untiled concatenation case.
  if (!tiling_dim.has_value()) {
    out_vregs = concatenate(operand_vregs, dimension);
  } else {
    bool is_rank1_with_no_implicit_dim = res_ty.getRank() == 1 &&
                                    res_layout->implicit_dim() ==
                                        VectorLayout::ImplicitDim::kNone;
    if (res_layout->implicit_dim() == VectorLayout::ImplicitDim::kMinor ||
        is_rank1_with_no_implicit_dim) {
      return op.emitOpError("Not implemented: implicit dim");
    }
    if (res_layout->implicit_dim() == VectorLayout::ImplicitDim::kSecondMinor &&
        res_layout->bitwidth() != 32) {
      return op.emitOpError(
          "Not implemented: only 32-bit bitwidth supported for SecondMinor "
          "implicit dim");
    }
    if (res_layout->offsets()[tiling_dim.value()] != 0) {
      return op.emitOpError("Not implemented: result non-zero offset.");
    }
    if (!res_layout->hasNativeTiling(ctx.target_shape) &&
        res_ty.getRank() != 1) {
      return op.emitOpError("Not implemented: Non native tiling in concat.");
    }

    int64_t offset_at_dim = 0;
    {
      for (int i = 0; i < op.getNumOperands(); ++i) {
        Value operand = op.getOperand(i);
        const Layout &layout = *layouts_in[i];
        xla::Array<Value> vreg_array = operand_vregs[i];
        std::array<int64_t, 2> vreg_slice = layout->vregSlice(ctx.target_shape);
        std::array<int64_t, 2> tiling = layout->tiling();

        VectorType vty = cast<VectorType>(operand.getType());
        ArrayRef<int64_t> shape = vty.getShape();

        int64_t starting_point = offset_at_dim;
        int64_t offset_amount =
            starting_point % vreg_slice[tiling_dim.value()];
        if (offset_amount >= tiling[tiling_dim.value()]) {
          return op.emitError(
              "Not implemented: Input offsets outside of the first tile");
        }
        if (offset_amount != layout->offsets()[tiling_dim.value()]) {
          return op.emitOpError(
              "Not implemented: Relayout not called, unaligned dims "
              "concatenated without proper offsets. Ensure that "
              "infer_vector_layout pass was called.");
        }
        offset_at_dim += shape[dimension];
      }
    }

    // Tiled concatenation logic.
    int64_t offset = 0;
    for (size_t i = 0; i < operand_vregs.size(); ++i) {
      auto &vreg = operand_vregs[i];
      const auto &layout = layouts_in[i];
      const int packing = res_layout->packing();

      if (layout->tiling()[0] % packing != 0) {
        return op.emitOpError(
            "Illegal tiling: Non-native tiling in concat - this should "
            "have been caught earlier!");
      }

      const int64_t operand_offset = *layout->offsets()[tiling_dim.value()];
      if (operand_offset != 0) {
        // We are offset, so we must blend with the previous vreg.
        // Or, to frame it in an another way, the prior vreg
        // stored its entire dim size in the offset, but only wrote the
        // last dime partially.
        offset -= 1;
      }

      const auto bitwidth = res_ty.getElementTypeBitWidth();
      SmallVector<int64_t> out_idx;
      vreg.Each([&](absl::Span<const int64_t> idx, Value *v) {
        out_idx.assign(idx.begin(), idx.end());
        out_idx[dimension] += offset;
        if (idx[dimension] == 0 && operand_offset != 0) {
          Value mask;
          const VectorType vmask_ty = getNativeVregOrVmaskType(
              builder.getI1Type(), bitwidth, ctx.target_shape);
          if (tiling_dim.value() == 0) {  // sublane
            if (operand_offset % packing != 0) {
              // Packed case, degenerate where we have a half or quarter
              // sublane.
              // TODO(mvoz): We can probably always use the
              // CreateSubelementMaskOp if (1) optimize it on TPUv4 and (2) Add
              // support for unpacked types in some of the invariants in
              // lower_to_llo.
              mask = builder.create<tpu::CreateSubelementMaskOp>(
                  op.getLoc(), vmask_ty, 0, operand_offset, packing);
            } else {
              auto sublane_offset = operand_offset / packing;
              mask = builder.create<tpu::CreateMaskOp>(
                  op.getLoc(), vmask_ty,
                  ArrayRef<Value>{boundIdxConst(0), boundIdxConst(0)},
                  ArrayRef<Value>{boundIdxConst(sublane_offset),
                                  boundIdxConst(layout->tiling()[1])});
            }
          } else {  // lane
            mask = builder.create<tpu::CreateMaskOp>(
                op.getLoc(), vmask_ty,
                ArrayRef<Value>{boundIdxConst(0), boundIdxConst(0)},
                ArrayRef<Value>{boundIdxConst(layout->tiling()[0] / packing),
                                boundIdxConst(operand_offset)});
          }
          // Blend the current value with the existing value in the output.
          *v = builder.create<arith::SelectOp>(op.getLoc(), mask,
                                               out_vregs(out_idx), *v);
        }
        out_vregs(out_idx) = *v;
      });
      offset += vreg.dim(dimension);
    }
  }
  auto assembled =
      assemble(builder, res_ty, *res_layout, out_vregs, ctx.target_shape);
  op.replaceAllUsesWith(assembled);
  op.erase();
  return success();
}

LogicalResult tpu_iota_rule(RewriteContext &ctx, Operation &op,
                            const ArrayRef<Layout> layouts_in,
                            const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  tpu::IotaOp iota_op = cast<tpu::IotaOp>(op);
  VectorType vty = iota_op.getResult().getType();
  if (const auto int_ty = dyn_cast<IntegerType>(vty.getElementType());
      int_ty == nullptr || int_ty.getWidth() != 32) {
    return iota_op.emitOpError("Not implemented: Only 32-bit Iota supported");
  }
  if (!layout_out.hasNativeTiling(ctx.target_shape)) {
    return iota_op.emitOpError("Not implemented: Only native tiling supported");
  }

  const auto native_vreg_ty =
      getNativeVregType(vty.getElementType(), ctx.target_shape);
  if (layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    return op.emitOpError("Not implemented: Only 2D layouts supported");
  }
  const SmallVector<int64_t> tile_array_shape =
      layout_out.tileArrayShape(vty.getShape(), ctx.target_shape);
  const std::optional<int32_t> dimension = iota_op.getDimension();
  if (!dimension.has_value()) {
    return op.emitOpError("Not implemented: null dimension");
  }
  if (*dimension == vty.getRank() - 1) {
    if (layout_out.offsets()[1] != 0) {
      return op.emitOpError("Not implemented: Unsupported offset");
    }
    const int64_t num_tiles = tile_array_shape[tile_array_shape.size() - 1];
    SmallVector<Value> tiles(num_tiles);
    auto vreg_iota = builder.create<tpu::IotaOp>(
        native_vreg_ty,
        /*dimension =*/builder.getI32IntegerAttr(1));
    for (int64_t i = 0; i < num_tiles; ++i) {
      Value offset = getFullVector(
          builder, native_vreg_ty,
          IntegerAttr::get(vty.getElementType(),
                           i * *(native_vreg_ty.getShape().end() - 1)));
      tiles[i] = builder.create<arith::AddIOp>(vreg_iota, offset);
    }
    xla::Array<Value> broadcasted_tiles(tile_array_shape);
    broadcasted_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      *v = tiles[*(idxs.end() - 1)];
    });
    op.replaceAllUsesWith(assemble(builder, vty, layout_out, broadcasted_tiles,
                                   ctx.target_shape));
    op.erase();
    return success();
  }
  if (*dimension == vty.getRank() - 2) {
    if (layout_out.offsets()[0] != 0) {
      return op.emitOpError("Not implemented: Unsupported offset");
    }
    const int64_t num_tiles = tile_array_shape[tile_array_shape.size() - 2];
    SmallVector<Value> tiles(num_tiles);
    auto vreg_iota = builder.create<tpu::IotaOp>(
        native_vreg_ty,
        /*dimension =*/builder.getI32IntegerAttr(0));
    for (int64_t i = 0; i < num_tiles; ++i) {
      Value offset = getFullVector(
          builder, native_vreg_ty,
          IntegerAttr::get(vty.getElementType(),
                           i * *(native_vreg_ty.getShape().end() - 2)));
      tiles[i] = builder.create<arith::AddIOp>(vreg_iota, offset);
    }
    xla::Array<Value> broadcasted_tiles(tile_array_shape);
    broadcasted_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      *v = tiles[*(idxs.end() - 2)];
    });
    op.replaceAllUsesWith(assemble(builder, vty, layout_out, broadcasted_tiles,
                                   ctx.target_shape));
    op.erase();
    return success();
  }
  // We take the iota over an untiled dimension.
  CHECK_LT(*dimension, vty.getRank());
  SmallVector<Value> tiles;
  tiles.reserve(vty.getDimSize(*dimension));
  for (int64_t i = 0; i < vty.getDimSize(*dimension); ++i) {
    tiles.push_back(getFullVector(builder, native_vreg_ty,
                                  IntegerAttr::get(vty.getElementType(), i)));
  }
  xla::Array<Value> out_tiles(tile_array_shape);
  out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
    *v = tiles[idxs[*dimension]];
  });
  op.replaceAllUsesWith(
      assemble(builder, vty, layout_out, out_tiles, ctx.target_shape));
  op.erase();
  return success();
}

LogicalResult tpu_gather_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_in.offsets() != layout_out.offsets() ||
      llvm::any_of(layout_in.offsets(), [&](const LayoutOffset o) {
        return o.has_value() && o != 0;
      })) {
    return op.emitOpError("Not implemented: Only 2D layouts supported");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto gather_op = cast<tpu::GatherOp>(op);
  const VectorType vty = gather_op.getResult().getType();
  const uint32_t dimension = gather_op.getDimension();
  if (dimension + 2 < vty.getRank()) {
    return op.emitOpError("Not implemented: Unsupported dimension");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> in_tiles,
      disassemble(builder, layout_in, gather_op.getSource(), ctx.target_shape));
  const int64_t width = ctx.target_shape[2 - (vty.getRank() - dimension)];
  const ArrayRef<int32_t> indices(gather_op.getIndices());
  auto [num_sections, rem] = std::div(indices.size(), width);
  SmallVector<int32_t> segment_indices;
  if (rem == 0) {
    for (int64_t i = 0; i < width; ++i) {
      const int64_t offset = i - i % width;
      if (!(offset <= indices[i] && indices[i] < offset + width)) {
        return op.emitOpError("Not implemented: Cross-segment gather");
      }
    }
    for (int64_t i = width; i < indices.size(); ++i) {
      const int64_t offset = i - i % width;
      if (indices[i] != indices[i % width] + offset) {
        return op.emitOpError(
            "Not implemented: Indices varying between segments");
      }
    }
    segment_indices.assign(indices.begin(), indices.begin() + width);
  } else if (num_sections == 0) {  // Only one vreg.
    segment_indices.assign(indices.begin(), indices.end());
    segment_indices.append(width - indices.size(), 0);
  } else {
    return op.emitOpError("Not implemented: Not a multiple of target length");
  }
  xla::Array<Value> out_tiles(in_tiles.dimensions());
  if (dimension == vty.getRank() - 1) {
    // TODO(b/265133497): Remove the broadcast once 2nd minor works.
    const auto dyn_ix_ty =
        VectorType::get(ctx.target_shape, builder.getI32Type());
    // Broadcast indices to target_shape
    SmallVector<int32_t> dyn_ix_val;
    for (int64_t i = 0; i < ctx.target_shape[0]; ++i) {  // Broadcast
      dyn_ix_val.append(segment_indices);
    }
    auto func_op = op.getParentOfType<func::FuncOp>();
    if (!func_op) {
      return op.emitOpError("Expected a function op");
    }
    FAILUREOR_ASSIGN_OR_RETURN(
        const BlockArgument dyn_ix_ref,
        appendConstant(ctx, func_op,
                       DenseIntElementsAttr::get(dyn_ix_ty, dyn_ix_val)));
    auto all_sublanes = builder.getAttr<DenseBoolArrayAttr>(
        SmallVector<bool>(ctx.target_shape[1], true));
    auto dyn_ix = builder.create<tpu::LoadOp>(
        dyn_ix_ty, dyn_ix_ref,
        SmallVector<Value>(2, IdxConst(0, builder, op.getLoc())),
        /*sublane_mask=*/all_sublanes, /*sublane_stride=*/nullptr);
    out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      const Value in_tile = in_tiles(idxs);
      *v = builder.create<tpu::DynamicGatherOp>(in_tile.getType(), in_tile,
                                                dyn_ix, 1);
    });
  } else {
    TPU_ASSERT_EQ_OP(dimension, vty.getRank() - 2);
    const auto segment_indices_attr =
        builder.getAttr<DenseI32ArrayAttr>(segment_indices);
    out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      const Value in_tile = in_tiles(idxs);
      *v = builder.create<tpu::GatherOp>(in_tile.getType(), in_tile,
                                         segment_indices_attr, 0);
    });
  }
  gather_op.replaceAllUsesWith(
      assemble(builder, vty, layout_out, out_tiles, ctx.target_shape)
          .getOperation());
  gather_op.erase();
  return success();
}

LogicalResult tpu_region_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  if (op.getNumOperands() != 0) {
    return op.emitOpError(
        "Not implemented: tpu.region_block with inputs");
  }
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto region_op = cast<tpu::RegionOp>(op);
  // We don't modify the op, but we do rewrite the branch bodies.
  if (failed(
          applyLayoutBlock(ctx, region_op.getRegion().getBlocks().front()))) {
    return op.emitOpError("Failed to apply layout to TPU region.");
  }
  auto yield_op = cast<tpu::YieldOp>(
      *region_op.getRegion().getBlocks().front().getTerminator());
  auto new_op = builder.create<tpu::RegionOp>(yield_op->getOperandTypes());
  moveAllRegions(*region_op, *new_op);

  int64_t index = 0;
  SmallVector<Value> rolled_results;
  for (auto [result, layout] :
       llvm::zip_equal(region_op.getResults(), layouts_out)) {
    if (const auto vty = dyn_cast<VectorType>(result.getType())) {
      // When the result has a vector type, assemble the result.
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(index + num_vectors, new_op.getResults().size());
      tiles.SetValues(
          llvm::make_range(new_op.getResults().begin() + index,
                           new_op.getResults().begin() + index + num_vectors));
      index += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      rolled_results.push_back(rolled_op);
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      rolled_results.push_back(new_op.getResult(index));
      ++index;
    }
  }
  region_op.replaceAllUsesWith(rolled_results);
  region_op.erase();
  return success();
}

LogicalResult vector_load_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  MLIRContext *const mlir_ctx = op.getContext();
  TPU_ASSERT_OP(llvm::none_of(layouts_in,
                              [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto load_op = cast<vector::LoadOp>(op);
  const auto memref_ty = getMemRefType(load_op.getBase());
  const auto vty = cast<VectorType>(load_op.getResult().getType());
  VectorType target_ty =
      getNativeVregType(vty.getElementType(), ctx.target_shape);
  if (vty.getRank() == 0) {
    op.emitOpError("Not implemented: scalar loads from vmem");
  }
  const bool is_1d = vty.getRank() == 1;
  VectorLayout::ImplicitDim expected_dim =
      is_1d ? VectorLayout::ImplicitDim::kSecondMinor
            : VectorLayout::ImplicitDim::kNone;
  if (layout_out.implicit_dim() != expected_dim) {
    return op.emitOpError("Not implemented: unsupported layout");
  }
  using Tiling = std::array<int64_t, 2>;  // To avoid comma in macro
  FAILUREOR_ASSIGN_OR_RETURN(
      Tiling memref_tiling,
      getMemRefTiling(load_op.getBase(), ctx.target_shape));
  if (memref_tiling != layout_out.tiling()) {
    if (memref_tiling[0] == 1 && layout_out.tiling()[0] == 1 &&
        memref_tiling[1] % layout_out.tiling()[1] == 0) {
      // In this case, it is valid to use output tiling (1, 128 * packing) when
      // loading from a 1D memref.
    } else if (layout_out.bitwidth() == 32 &&
               layout_out.tiling() ==
                   std::array<int64_t, 2>{1, ctx.target_shape[1]}) {
      // In this case, it is valid to use output tiling (1, TARGET_SHAPE.lanes)
      // because we strided-load one row from each tile of the memref. This can
      // save us a bunch of loads!
      // TODO(b/295393167): need to support strided load for bitwidth < 32.
    } else if (layout_out.bitwidth() == 32 &&
               canReinterpretToUntiledMemref(
                   load_op.getBase(), ctx.target_shape,
                   /*allow_minormost_padding=*/true)) {
      // In this case, if the memref can be reinterpreted to untiled, it is
      // valid to use any tiling for output. But using native tiling can save us
      // a bunch of loads!
    } else {
      return op.emitOpError(
          "Not implemented: dismatch in memref tiling and vector tiling in "
          "load");
    }
  }
  // TODO(apaszke): Check that loads are from vmem!

  bool can_support_unaligned_dynamic_index = false;
  bool must_support_unaligned_dynamic_index = false;
  if (load_op.getIndices().size() > 1) {
    auto second_minor_idx = load_op.getIndices().take_back(2)[0];
    if (failed(getIntConst(second_minor_idx, /*silent=*/true)) &&
        !isGuaranteedDivisible(second_minor_idx, memref_tiling[0])) {
      must_support_unaligned_dynamic_index = true;
    }
  }
  const SmallVector<int64_t> implicit_shape =
      layout_out.implicitShape(vty.getShape());
  const int64_t ss = implicit_shape[implicit_shape.size() - 2];
  int64_t sublane_stride = 1;
  // Handle special patterns that allow us to support more flexible loads.
  if (layout_out.bitwidth() == 32 &&
      layout_out.tiling() == std::array<int64_t, 2>{1, ctx.target_shape[1]} &&
      ss == 1) {
    // Loading a single row on the 2nd minor dim into the (1, 128) layout. We
    // can use sublane striding to perform the relayout as part of the load.
    sublane_stride = memref_tiling[0];
    can_support_unaligned_dynamic_index = true;
  } else {
    // Otherwise, if the memref has a short last dimension and is contiguous
    // all the tiled layouts become equivalent, so we can handle unaligned
    // dynamic indices without any special case.
    auto mem_layout = dyn_cast<TiledLayoutAttr>(memref_ty.getLayout());
    if (!mem_layout) {
      return op.emitOpError("Expected a tiled memref");
    }
    auto tile_strides = mem_layout.getTileStrides();
    if (memref_ty.getShape().back() == ctx.target_shape[1] &&
        tile_strides.take_back(2) == ArrayRef<int64_t>{1, 1}) {
      can_support_unaligned_dynamic_index = true;
    }
  }

  auto add_idx = [&](const Value &v, int64_t d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return IdxConst(cst.value() + d, builder, op.getLoc());
    }
    return builder.create<arith::AddIOp>(v, IdxConst(d, builder, op.getLoc()));
  };

  int tiled_dims = is_1d ? 1 : 2;
  Value base_addr = load_op.getBase();
  SmallVector<Value, 4> base_indices = load_op.getIndices();

  if (must_support_unaligned_dynamic_index) {
    if (!can_support_unaligned_dynamic_index) {
      return op.emitOpError(
          "Not implemented: dynamic load with unaligned indices");
    }
  } else {
    // Convert dynamic load to dynamic slice + static load. This saves us a
    // bunch of scalar core work.
    auto slice_result =
        sliceRef(builder, load_op.getBase(), load_op.getVectorType().getShape(),
                 load_op.getIndices(),
                 ArrayRef<int64_t>(memref_tiling).take_back(tiled_dims));
    if (failed(slice_result)) {
      return failure();
    }
    base_addr = slice_result->first;
    CHECK_EQ(slice_result->second.size(), base_indices.size());
    for (int i = 0; i < base_indices.size(); ++i) {
      base_indices[i] = IdxConst(slice_result->second[i], builder, op.getLoc());
    }
  }

  // TODO(jevinjiang): ideally we should update the base addr and use static
  // indices even for the cases that can skip alignment check. This can save us
  // a bunch of scalar core work.
  auto tile_base_idxs = ArrayRef<Value>(base_indices).take_back(tiled_dims);
  auto batch_base_idxs = ArrayRef<Value>(base_indices).drop_back(tiled_dims);
  const LayoutOffsets offsets = layout_out.offsets();
  AffineMap load_map;
  if (offsets[1] == std::nullopt) {
    return op.emitOpError(
        "Not implemented: Load replicated along lanes is unsupported");
  }
  if (offsets[0] == std::nullopt) {
    if (ss != 1) {
      return op.emitOpError(
          "Not implemented: Sublane-replicated load with size > 1 is "
          "unsupported");
    }
    if (!layout_out.hasNativeTiling(ctx.target_shape)) {
      return op.emitOpError("Not implemented");
    }
    // affine_map<(..., j) -> (0, j)
    load_map =
        AffineMap::get(memref_ty.getRank(), 0,
                       {getAffineConstantExpr(0, mlir_ctx),
                        getAffineDimExpr(memref_ty.getRank() - 1, mlir_ctx)},
                       mlir_ctx);
  }

  xla::Array<Value> tiles(
      layout_out.tileArrayShape(vty.getShape(), ctx.target_shape));
  const std::array<int64_t, 2> vreg_slice =
      layout_out.vregSlice(ctx.target_shape);
  const int64_t num_dims = vty.getRank();
  const int64_t num_batch_dims = num_dims - (is_1d ? 1 : 2);
  const absl::Status status =
      tiles.EachStatus([&](absl::Span<const int64_t> tile_idxs, Value * /*v*/) {
        CHECK_EQ(num_dims, tile_idxs.size());
        SmallVector<Value> idxs(tile_idxs.size());
        for (int64_t i = 0; i < num_batch_dims; ++i) {
          idxs[i] = add_idx(batch_base_idxs[i], tile_idxs[i]);
        }
        const auto base_l = tile_base_idxs.back();
        const int64_t lidx = tile_idxs[num_dims - 1];
        idxs[num_dims - 1] =
            add_idx(base_l, lidx * vreg_slice[1] - offsets[1].value_or(0));
        if (!is_1d) {
          CHECK_EQ(tile_base_idxs.size(), 2);
          const auto base_s = tile_base_idxs.front();
          const int64_t sidx = tile_idxs[num_dims - 2];
          idxs[num_dims - 2] =
              add_idx(base_s, sidx * vreg_slice[0] - offsets[0].value_or(0));
        }
        TPU_ASSERT_OP(tile_idxs[num_dims - 1] + ctx.target_shape[1] <=
                      memref_ty.getShape()[num_dims - 1]);
        std::unique_ptr<VRegDataBounds> bounds = layout_out.tileDataBounds(
            mlir_ctx, vty.getShape(), toArrayRef(tile_idxs), ctx.target_shape,
            /*allow_replicated =*/{true, false});
        Operation *tile;
        if (bounds->maskVariesAlong(Direction::kSublanes, ctx.target_shape)) {
          CHECK(offsets[0].has_value());
          tile = builder.create<tpu::LoadOp>(
              target_ty, base_addr, idxs,
              bounds->getSublaneMask(mlir_ctx, ctx.target_shape),
              builder.getI32IntegerAttr(sublane_stride));
        } else {
          if (load_map) {
            if (layout_out.bitwidth() != 32) {
              load_op.emitOpError("Not implemented");
              return absl::UnimplementedError("");
            }
            tile = builder.create<vector::TransferReadOp>(
                target_ty, base_addr, idxs, load_map,
                // TODO(tlongeri): Not sure whether we are obeying the semantics
                // of in_bounds, but our lowering ignores it and this path will
                // removed soon anyway.
                SmallVector<bool>(2, true));
          } else {
            const SmallVector<bool> sublane_mask(ctx.target_shape[0], true);
            const auto sublane_mask_attr =
                DenseBoolArrayAttr::get(mlir_ctx, sublane_mask);
            tile = builder.create<tpu::LoadOp>(
                target_ty, base_addr, idxs, sublane_mask_attr,
                builder.getI32IntegerAttr(sublane_stride));
          }
        }
        tiles(tile_idxs) = tile->getResult(0);
        return absl::OkStatus();
      });
  if (!status.ok()) {
    return failure();
  }
  load_op->replaceAllUsesWith(
      assemble(builder, vty, layout_out, std::move(tiles), ctx.target_shape));
  load_op->erase();
  return success();
}

LogicalResult arith_constant_rule(RewriteContext &ctx, Operation &op,
                                  const ArrayRef<Layout> layouts_in,
                                  const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto constant_op = cast<arith::ConstantOp>(op);
  auto vty = dyn_cast<VectorType>(op.getResult(0).getType());
  if (vty) {
    if (!layouts_out.front().has_value()) {
      return op.emitOpError(
          "Expected non-null output layout for vector constant");
    }
    const VectorLayout &layout_out = *layouts_out.front();
    DenseElementsAttr value = cast<DenseElementsAttr>(constant_op.getValue());
    const VectorType target_vty = getNativeVregOrVmaskType(
        vty.getElementType(), layout_out.bitwidth(), ctx.target_shape);
    if (value.isSplat()) {
      if (layout_out.offsets() != LayoutOffsets{std::nullopt, std::nullopt}) {
        return op.emitOpError(
            "Not implemented: Non-replicated splat constants");
      }
      auto new_value =
          DenseElementsAttr::get(target_vty, value.getSplatValue<Attribute>());
      const auto tile =
          builder.create<arith::ConstantOp>(target_vty, new_value);
      const xla::Array<Value> tiles(
          layout_out.tileArrayShape(vty.getShape(), ctx.target_shape),
          tile->getResult(0));
      op.replaceAllUsesWith(assemble(builder, vty, layout_out, std::move(tiles),
                                     ctx.target_shape));
      op.erase();
      return success();
    }
    // !value.isSplat()
    if (getTypeBitwidth<true>(vty.getElementType()) != 32) {
      return op.emitOpError(
          "Not implemented: Only 32-bit non-splat constants are supported");
    }
    auto func_op = op.getParentOfType<func::FuncOp>();
    if (!func_op) {
      return op.emitOpError("Expected a function op");
    }
    FAILUREOR_ASSIGN_OR_RETURN(const BlockArgument ref,
                               appendConstant(ctx, func_op, value));
    auto load_op = builder.create<vector::LoadOp>(
        vty, ref,
        SmallVector<Value>(vty.getRank(), IdxConst(0, builder, op.getLoc())));
    op.replaceAllUsesWith(ArrayRef<Value>{load_op.getResult()});
    op.erase();
    const SmallVector<Layout> vector_load_in_layouts(vty.getRank() + 1);
    return vector_load_rule(ctx, *load_op, vector_load_in_layouts,
                            {VectorLayout(/*bitwidth=*/32, /*offsets=*/{0, 0},
                                          /*tiling=*/ctx.target_shape)});
  }
  return op.emitOpError("Not implemented: Unsupported arith.const type: ")
         << op.getResult(0).getType();
}

LogicalResult vector_broadcast_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const Layout &maybe_layout_in = layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  vector::BroadcastOp broadcast_op = cast<vector::BroadcastOp>(op);
  const VectorType dst_ty = broadcast_op.getResult().getType();
  const ArrayRef<int64_t> dst_shape = dst_ty.getShape();
  const SmallVector<int64_t> dst_tiles_shape =
      layout_out.tileArrayShape(dst_shape, ctx.target_shape);
  const SmallVector<int64_t> dst_tiles_implicit_shape =
      layout_out.tileArrayImplicitShape(dst_shape, ctx.target_shape);
  if (auto src = dyn_cast<TypedValue<VectorType>>(broadcast_op.getSource())) {
    VectorType src_ty = src.getType();
    TPU_ASSERT_OP(maybe_layout_in.has_value());
    const VectorLayout &layout_in = *maybe_layout_in;
    if (layout_in.implicit_dim() != layout_out.implicit_dim()) {
      return op.emitOpError(
          "Not implemented: Changing implicit dims mid-broadcast");
    }
    const LayoutOffsets offsets_in = layout_in.offsets();
    const LayoutOffsets offsets_out = layout_out.offsets();
    if (layout_in.tiling() != layout_out.tiling()) {
      return op.emitOpError("Not implemented: Changing tiling mid-broadcast");
    }
    auto tiling = layout_in.tiling();

    const int64_t expand_rank = dst_ty.getRank() - src_ty.getRank();
    const ArrayRef<int64_t> src_shape = src_ty.getShape();

    SmallVector<int64_t> src_implicit_shape_padded;
    // `is_logical_broadcast` stores whether each dimension of the implicit
    // shape of the result is a broadcast. E.g. if the implicit shape goes from
    // (2, 1, 3) to (4, 2, 5, 3) it's (true, false, true, false).
    SmallVector<bool> is_logical_broadcast;
    src_implicit_shape_padded.reserve(dst_shape.size() +
                                      layout_in.num_implicit_dims());
    is_logical_broadcast.reserve(dst_shape.size() +
                                 layout_in.num_implicit_dims());
    src_implicit_shape_padded.append(expand_rank, 1);
    src_implicit_shape_padded.append(src_shape.begin(), src_shape.end());
    for (auto [i, o] : llvm::zip(src_implicit_shape_padded, dst_shape)) {
      TPU_ASSERT_OP(i == o || i == 1);  // Verifier should guarantee this.
      is_logical_broadcast.push_back(i != o);
    }
    layout_in.insertImplicit<int64_t>(src_implicit_shape_padded, 1);
    layout_in.insertImplicit<bool>(is_logical_broadcast, false);

    // Verify that the offsets are valid.
    for (auto [is_logical_broadcast_on_dim, in_off, out_off] :
         llvm::zip_equal(ArrayRef(is_logical_broadcast).take_back(2),
                         offsets_in, offsets_out)) {
      if (is_logical_broadcast_on_dim) {
        if (out_off.has_value()) {
          // There's no reason to ever assign a non-replicated offset to a
          // broadcasted dimension in the output.
          return op.emitOpError(
              // TODO(tlongeri): This should never be implemented but the fuzzed
              //                 tests expect a NotImplementedError, which
              //                 is raised with a "Not implemented" (see
              //                 NotImplementedDetector in tpu_ext.cc). Fix.
              "Not implemented: Broadcast output expected to have replicated "
              "offsets.");
        }
      } else {  // !is_logical_broadcast_on_dim
        if (in_off != out_off) {
          return op.emitOpError(
              "Not implemented: Changing offsets mid-broadcast");
        }
      }
    }

    // `needs_physical_broadcast` specifies whether we need to broadcast vregs
    // vregs in the sublane and lane dimensions. We only need to do this if the
    // corresponding dimension of the implicit shape is logically broadcast and
    // if the input vregs are not already replicated along this dimension.
    const std::array<bool, 2> needs_physical_broadcast{
        *(is_logical_broadcast.end() - 2) && offsets_in[0].has_value(),
        *(is_logical_broadcast.end() - 1) && offsets_in[1].has_value()};
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> src_tiles,
        disassemble(builder, layout_in, src, ctx.target_shape,
                    /*use_implicit_shape=*/true));
    xla::Array<Value> dst_tiles(dst_tiles_implicit_shape);
    if (needs_physical_broadcast == std::array{false, false}) {  // No-op
      SmallVector<int64_t> reshape_dims(expand_rank, 1);
      const absl::Span<const int64_t> src_tiles_dims = src_tiles.dimensions();
      reshape_dims.append(src_tiles_dims.begin(), src_tiles_dims.end());
      src_tiles.Reshape(reshape_dims);
      dst_tiles.Each([&](const absl::Span<const int64_t> dst_idx, Value *tile) {
        const SmallVector<int64_t> src_idx = llvm::map_to_vector(
            llvm::zip_equal(dst_idx, is_logical_broadcast), [](auto tup) {
              auto [i, is_logical_broadcast_on_dim] = tup;
              return is_logical_broadcast_on_dim ? 0 : i;
            });
        *tile = src_tiles(src_idx);
      });
    } else {
      if (tiling[1] != ctx.target_shape[1]) {
        return op.emitOpError("Not implemented: unsupported tiling");
      }
      int64_t num_tiles = layout_in.tilesPerVreg(ctx.target_shape);
      if (needs_physical_broadcast ==
          std::array{true, false}) {  // Sublane broadcast
        const int packing = layout_in.packing();
        if (num_tiles != 1) {
          return op.emitOpError(
              "Not implemented: Only native tiling supported");
        }
        TPU_ASSERT_EQ_OP(*(src_tiles.dimensions().end() - 2), 1);
        TPU_ASSERT_OP(offsets_in[0].has_value());
        const int64_t sublane_offset = *offsets_in[0] / packing;
        const int64_t subelement_offset = *offsets_in[0] % packing;
        const DenseI32ArrayAttr indices = builder.getDenseI32ArrayAttr(
            SmallVector<int32_t>(ctx.target_shape[0], sublane_offset));
        const absl::Status status =
            src_tiles.EachStatus([&](const absl::Span<const int64_t> src_idx,
                                     Value *const src_vreg) {
              Value dst_vreg = *src_vreg;
              // Replicate the value within each sublane.
              if (packing != 1) {
                if (auto new_dst_vreg = broadcastSubelements(
                        builder, cast<TypedValue<VectorType>>(dst_vreg),
                        subelement_offset, ctx.target_shape,
                        ctx.hardware_generation);
                    succeeded(new_dst_vreg)) {
                  dst_vreg = *new_dst_vreg;
                } else {
                  return absl::InternalError("");
                }
              }
              dst_vreg = builder.create<tpu::GatherOp>(dst_vreg.getType(),
                                                       dst_vreg, indices, 0);
              SmallVector<int64_t> dst_starts(dst_tiles_implicit_shape.size());
              SmallVector<int64_t> dst_limits(dst_tiles_implicit_shape.size());
              for (int64_t i = 0; i < dst_tiles.num_dimensions(); ++i) {
                if (i < expand_rank || is_logical_broadcast[i]) {
                  dst_starts[i] = 0;
                  dst_limits[i] = dst_tiles_implicit_shape[i];
                } else {
                  dst_starts[i] = src_idx[i - expand_rank];
                  dst_limits[i] = dst_starts[i] + 1;
                }
              }
              updateSlice<Value>(dst_tiles, dst_vreg, dst_starts, dst_limits);
              return absl::OkStatus();
            });
        if (!status.ok()) {
          return failure();
        }
      } else if (needs_physical_broadcast ==
                 std::array{false, true}) {  // Lane broadcast
        TPU_ASSERT_EQ_OP(*(src_tiles.dimensions().end() - 1), 1);
        TPU_ASSERT_OP(offsets_in[1].has_value());
        const int64_t sublanes_per_tile =
            layout_in.sublanesPerTile(ctx.target_shape);
        const int64_t offset = *offsets_in[1];
        const int64_t lane_offset = offset % ctx.target_shape[1];
        const int64_t tile_offset = offset / ctx.target_shape[1];
        Value lane_offset_cst = getFullVector(
            builder, getNativeVregType(builder.getI32Type(), ctx.target_shape),
            builder.getI32IntegerAttr(lane_offset));
        DenseI32ArrayAttr sublane_pattern;
        if (num_tiles != 1) {
          SmallVector<int32_t> pattern;
          pattern.reserve(ctx.target_shape[0]);
          for (int32_t t = 0; t < num_tiles; ++t) {
            for (int32_t i = 0; i < sublanes_per_tile; ++i) {
              pattern.push_back(sublanes_per_tile * tile_offset + i);
            }
          }
          sublane_pattern = builder.getDenseI32ArrayAttr(pattern);
        }
        src_tiles.Each([&](const absl::Span<const int64_t> src_idx,
                           Value *const src_tile) {
          SmallVector<int64_t> dst_starts(dst_tiles_implicit_shape.size());
          SmallVector<int64_t> dst_limits(dst_tiles_implicit_shape.size());
          for (int64_t i = 0; i < dst_tiles.num_dimensions(); ++i) {
            if (i < expand_rank || is_logical_broadcast[i]) {
              dst_starts[i] = 0;
              dst_limits[i] = dst_tiles_implicit_shape[i];
            } else {
              dst_starts[i] = src_idx[i - expand_rank];
              dst_limits[i] = dst_starts[i] + 1;
            }
          }
          Value res_vreg = builder.create<tpu::DynamicGatherOp>(
              broadcast_op.getLoc(), src_tile->getType(), *src_tile,
              lane_offset_cst,
              /*dimension=*/1);
          if (num_tiles != 1) {
            res_vreg = builder.create<tpu::GatherOp>(
                broadcast_op.getLoc(), res_vreg.getType(), res_vreg,
                sublane_pattern, 0);
          }
          updateSlice<Value>(dst_tiles, res_vreg, dst_starts, dst_limits);
        });
      } else {
        TPU_ASSERT_OP((needs_physical_broadcast == std::array{true, true}));
        return op.emitOpError(
            "Not implemented: Broadcast in both sublanes and lanes");
      }
    }
    broadcast_op.replaceAllUsesWith(assemble(builder, dst_ty, layout_out,
                                             dst_tiles, ctx.target_shape,
                                             /*use_implicit_shape=*/true)
                                        .getOperation());
    broadcast_op.erase();
    return success();
  } else if (layout_out.bitwidth() == 32 &&
             broadcast_op.getSourceType().getIntOrFloatBitWidth() == 1) {
    // Broadcasting the i1 scalar involves first converting i1 to i32, followed
    // by broadcasting i32 to the target shape. Finally, the comparison with 0s
    // yields the vmask.
    auto src_i32 = builder.create<arith::ExtUIOp>(
        broadcast_op.getLoc(), builder.getI32Type(), broadcast_op.getSource());

    const VectorType native_vreg_ty =
        getNativeVregType(src_i32.getType(), ctx.target_shape);
    auto tile_i32 =
        builder.create<vector::BroadcastOp>(native_vreg_ty, src_i32);
    Value zeros = getZerosVector(builder, tile_i32.getType());
    auto tile =
        builder.create<arith::CmpIOp>(arith::CmpIPredicate::ne, tile_i32, zeros)
            .getResult();
    const xla::Array<Value> dst_tiles(dst_tiles_shape, tile);
    broadcast_op.replaceAllUsesWith(
        assemble(builder, dst_ty, layout_out, dst_tiles, ctx.target_shape)
            .getOperation());
    broadcast_op.erase();
    return success();
  } else if (layout_out.bitwidth() < 32) {
    CHECK_EQ(layout_out.bitwidth(),
             broadcast_op.getSourceType().getIntOrFloatBitWidth());
    // Broadcasting the scalar with narrower type involves first packing (32 /
    // bitwidth) copies to i32, followed by broadcasting i32 to the target
    // shape. Finally, bitcast i32 vector back to the original narrower type
    // vector.
    auto loc = broadcast_op.getLoc();
    auto src_ty = broadcast_op.getSourceType();
    auto bitwidth = src_ty.getIntOrFloatBitWidth();
    auto unpacked_src = broadcast_op.getSource();
    if (!src_ty.isSignlessInteger(bitwidth)) {
      unpacked_src = builder.create<arith::BitcastOp>(
          loc, builder.getIntegerType(bitwidth), unpacked_src);
    }
    auto src_i32 =
        builder.create<arith::ExtUIOp>(loc, builder.getI32Type(), unpacked_src)
            .getResult();
    for (int i = 1; i < (32 / bitwidth); ++i) {
      auto shift_width = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(builder.getI32Type(), i * bitwidth));
      src_i32 = builder.create<arith::OrIOp>(
          loc, src_i32,
          builder.create<arith::ShLIOp>(loc, src_i32, shift_width));
    }

    const VectorType i32_vreg_ty =
        getNativeVregType(src_i32.getType(), ctx.target_shape);
    auto tile_i32 = builder.create<vector::BroadcastOp>(i32_vreg_ty, src_i32);

    const VectorType native_vreg_ty =
        getNativeVregType(src_ty, ctx.target_shape);
    auto tile = builder.create<tpu::BitcastVregOp>(native_vreg_ty, tile_i32);

    const xla::Array<Value> dst_tiles(dst_tiles_shape, tile);
    broadcast_op.replaceAllUsesWith(
        assemble(builder, dst_ty, layout_out, dst_tiles, ctx.target_shape)
            .getOperation());
    broadcast_op.erase();
    return success();
  } else {
    const VectorType native_vreg_ty =
        getNativeVregType(broadcast_op.getSourceType(), ctx.target_shape);
    auto tile = builder.create<vector::BroadcastOp>(native_vreg_ty,
                                                    broadcast_op.getSource());
    const xla::Array<Value> dst_tiles(dst_tiles_shape, tile);
    broadcast_op.replaceAllUsesWith(
        assemble(builder, dst_ty, layout_out, dst_tiles, ctx.target_shape)
            .getOperation());
    broadcast_op.erase();
    return success();
  }
}

// Returns slice of vregs containing a given slice of elements, obtained from
// the result of a vector.extract or vector.extract_strided_slice op.
//
// Takes offsets and sizes describing the slice of elements. If their size is
// less than the rank of the input vector, they describe a prefix i.e. they
// apply to the first (majormost) dimensions and the remaining dimensions are
// not sliced.
//
// Args:
// - ctx:        Rewrite context (for disassembling, which may create an op).
// - op:         Source vector.extract or vector.extract_strided_slice op.
// - offsets:    Prefix of offsets of slice of elements. Must have the same size
//               as sizes.
// - sizes:      Prefix of sizes of slice of elements. Must have the same size
//               as offsets.
// - layout_in:  Layout of src_vector.
// - layout_out: Layout that will be used to reassemble the slice (by caller).
//               Used only to check that the reassembling is valid.
FailureOr<xla::Array<Value>> vector_extract_slice_impl(
    RewriteContext &ctx, Operation &op, const ArrayRef<int64_t> sizes,
    const ArrayRef<int64_t> offsets, const VectorLayout &layout_in,
    const VectorLayout &layout_out) {
  if (layout_in.tiling() != layout_out.tiling() ||
      layout_in.bitwidth() != layout_out.bitwidth()) {
    return op.emitOpError(
        "Not implemented: Expected layout_in and layout_out tiling and packing "
        "to match");
  }

  // Both extract_strided_slice and extract have their input vector at index 0
  // and a single result.
  CHECK((isa<vector::ExtractOp, vector::ExtractStridedSliceOp>(op)));
  auto src_vector = cast<TypedValue<VectorType>>(op.getOperand(0));
  auto result = cast<TypedValue<VectorType>>(op.getResult(0));

  const VectorType dst_ty = result.getType();
  if (layout_in.implicit_dim() != layout_out.implicit_dim() &&
      !(layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
        layout_out.implicit_dim() == VectorLayout::ImplicitDim::kSecondMinor &&
        dst_ty.getRank() == 1)) {
    return op.emitOpError(
        "Not implemented: Unexpected change in implicit dimension that may not "
        "be a no-op");
  }

  const ArrayRef<int64_t> src_vector_shape = src_vector.getType().getShape();
  const int64_t src_vector_rank = src_vector_shape.size();
  const int64_t num_indices = offsets.size();
  TPU_ASSERT_EQ_OP(num_indices, sizes.size());

  SmallVector<int64_t> full_sizes;
  full_sizes.reserve(src_vector_rank + layout_in.num_implicit_dims());
  full_sizes.append(sizes.begin(), sizes.end());
  full_sizes.append(src_vector_shape.begin() + num_indices,
                    src_vector_shape.end());
  layout_in.insertImplicit<int64_t>(full_sizes, 1);

  SmallVector<int64_t> full_offsets;
  full_offsets.reserve(src_vector_rank + layout_in.num_implicit_dims());
  full_offsets.append(offsets.begin(), offsets.end());
  full_offsets.append(src_vector_rank - num_indices, 0);
  layout_in.insertImplicit<int64_t>(full_offsets, 0);

  // We currently only support no-op cases - that is, those where we effectively
  // just extract a slice of vregs without doing any operations (e.g. shifts) on
  // them.
  for (auto [index_offset, in_offset, vreg_slice, out_offset] : llvm::zip_equal(
           ArrayRef<int64_t>(full_offsets).take_back(2), layout_in.offsets(),
           layout_in.vregSlice(ctx.target_shape), layout_out.offsets())) {
    if (in_offset.has_value() != out_offset.has_value()) {
      return op.emitOpError(
          "Unexpected mismatch in replication between input and output "
          "layouts");
    }
    if (in_offset.has_value() &&
        (index_offset + *in_offset) % vreg_slice != *out_offset) {
      return op.emitOpError("Not implemented: Only no-op tiles");
    }
  }

  const std::array<int64_t, 2> vreg_slice =
      layout_in.vregSlice(ctx.target_shape);
  SmallVector<int64_t> slice_tiled_starts(full_offsets);
  *(slice_tiled_starts.end() - 2) =
      (layout_in.offsets()[0].value_or(0) + *(full_offsets.end() - 2)) /
      vreg_slice[0];
  *(slice_tiled_starts.end() - 1) =
      (layout_in.offsets()[1].value_or(0) + *(full_offsets.end() - 1)) /
      vreg_slice[1];
  layout_in.eraseImplicit(slice_tiled_starts);
  SmallVector<int64_t> slice_tiled_limits(full_offsets);
  for (int64_t i = 0; i < full_offsets.size() - layout_in.layout_rank(); ++i) {
    slice_tiled_limits[i] += full_sizes[i];
  }
  *(slice_tiled_limits.end() - 2) =
      llvm::divideCeil(layout_in.offsets()[0].value_or(0) +
                           *(full_offsets.end() - 2) + *(full_sizes.end() - 2),
                       vreg_slice[0]);
  *(slice_tiled_limits.end() - 1) =
      llvm::divideCeil(layout_in.offsets()[1].value_or(0) +
                           *(full_offsets.end() - 1) + *(full_sizes.end() - 1),
                       vreg_slice[1]);
  layout_in.eraseImplicit(slice_tiled_limits);

  OpBuilder builder(&op);
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> input_tiles,
      disassemble(builder, layout_in, src_vector, ctx.target_shape));
  return input_tiles.Slice(slice_tiled_starts, slice_tiled_limits);
}

LogicalResult vector_extract_rule(RewriteContext &ctx, Operation &op,
                                  const ArrayRef<Layout> layouts_in,
                                  const ArrayRef<Layout> layouts_out) {
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  vector::ExtractOp extract_op = cast<vector::ExtractOp>(op);
  if (extract_op.hasDynamicPosition()) {
    return op.emitOpError("Not implemented: dynamic indices");
  }
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  if (layout_in.bitwidth() != 32) {
    return op.emitOpError(
        "Not implemented: Only 32-bit vector.extract supported");
  }
  const VectorType res_vty =
      dyn_cast<VectorType>(extract_op.getResult().getType());
  if (res_vty != nullptr) {
    TPU_ASSERT_OP(layouts_out.front().has_value());
    const VectorLayout &layout_out = *layouts_out.front();
    const int64_t num_indices = extract_op.getStaticPosition().size();
    const SmallVector<int64_t> sizes(num_indices, 1);
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> dst_vregs,
        vector_extract_slice_impl(ctx, *extract_op, sizes,
                                  extract_op.getStaticPosition(), layout_in,
                                  *layouts_out.front()));
    // Squeeze leading singleton dimensions.
    TPU_ASSERT_EQ_OP(res_vty.getRank(),
                     extract_op.getSourceVectorType().getRank() - num_indices);
    TPU_ASSERT_OP(
        llvm::all_of(toArrayRef(dst_vregs.dimensions()).take_front(num_indices),
                     [](const int64_t d) { return d == 1; }));
    // Copy dims to temporary before passing to xla::Array::Reshape - it cannot
    // take a pointer to its own data.
    dst_vregs.Reshape(SmallVector<int64_t>(
        toArrayRef(dst_vregs.dimensions()).drop_front(num_indices)));
    op.replaceAllUsesWith(
        assemble(builder, res_vty, layout_out, dst_vregs, ctx.target_shape)
            .getOperation());
    op.erase();
    return success();
  } else {
    // TODO(b/367459476): Support non-zero offsets.
    if (layout_in.offsets() != LayoutOffsets{0, 0}) {
      return op.emitOpError("Not implemented: Unsupported layout");
    }
    auto [sub_tile, lane_tile] = layout_in.tiling();
    FAILUREOR_ASSIGN_OR_RETURN(
        const xla::Array<Value> vregs,
        disassemble(builder, layout_in, extract_op.getVector(),
                    ctx.target_shape));
    TPU_ASSERT_GT_OP(vregs.num_elements(), 0);

    SmallVector<int64_t> indices(extract_op.getStaticPosition());
    auto vreg_slice = layout_in.vregSlice(ctx.target_shape);
    std::array<int64_t, 2> position = {0, 0};
    SmallVector<int64_t> vreg_index(indices);
    // TODO(b/367459476): Support non-VREG-aligned tiling.
    CHECK_EQ(lane_tile, ctx.target_shape[1]);
    layout_in.insertImplicit(indices, static_cast<int64_t>(0));
    layout_in.insertImplicit(vreg_index, static_cast<int64_t>(0));
    int i = *(indices.end()-2);
    int j = *(indices.end()-1);
    *(vreg_index.end() -2) = i / vreg_slice[0];
    *(vreg_index.end() -1) = j / vreg_slice[1];
    layout_in.eraseImplicit(vreg_index);
    position[0] = ((j % vreg_slice[1]) / lane_tile * sub_tile
      ) + i % sub_tile;
    position[1] = j % lane_tile;

    TPU_ASSERT_LT_OP(vreg_index, vregs.dimensions());
    Value extracted_vreg = vregs(vreg_index);

    // Invert the offsets to get the rotation amount.
    position[0] = (ctx.target_shape[0] - position[0]) % ctx.target_shape[0];
    position[1] = (ctx.target_shape[1] - position[1]) % ctx.target_shape[1];
    auto res_vreg_ty = extracted_vreg.getType();
    Value shift = builder.create<arith::ConstantOp>(
        builder.getIntegerAttr(builder.getI32Type(), position[0]));
    Value rotated_vreg = builder.create<tpu::DynamicRotateOp>(
        res_vreg_ty, extracted_vreg, shift, 0, /*stride*/nullptr, nullptr);
    shift = builder.create<arith::ConstantOp>(
        builder.getIntegerAttr(builder.getI32Type(), position[1]));
    rotated_vreg = builder.create<tpu::DynamicRotateOp>(
        res_vreg_ty, rotated_vreg, shift, 1, /*stride*/nullptr, nullptr);
    extract_op.replaceAllUsesWith(
        builder.create<vector::ExtractOp>(
                op.getLoc(), rotated_vreg,
                ArrayRef<int64_t>{0, 0})
            .getResult());
  }
  extract_op.erase();
  return success();
}

LogicalResult vector_extract_strided_slice_rule(
    RewriteContext &ctx, Operation &op, const ArrayRef<Layout> layouts_in,
    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto extract_strided_slice_op = cast<vector::ExtractStridedSliceOp>(op);

  auto I64ArrayToSmallVector = [&](const ArrayAttr array_attr) {
    return llvm::map_to_vector(array_attr, [](Attribute attr) {
      return cast<IntegerAttr>(attr).getValue().getSExtValue();
    });
  };

  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> dst_vregs,
      vector_extract_slice_impl(
          ctx, *extract_strided_slice_op,
          I64ArrayToSmallVector(extract_strided_slice_op.getSizes()),
          I64ArrayToSmallVector(extract_strided_slice_op.getOffsets()),
          layout_in, layout_out));
  op.replaceAllUsesWith(assemble(builder,
                                 extract_strided_slice_op.getResult().getType(),
                                 layout_out, dst_vregs, ctx.target_shape)
                            .getOperation());
  op.erase();
  return success();
}

LogicalResult vector_multi_reduction_rule(RewriteContext &ctx, Operation &op,
                                          const ArrayRef<Layout> layouts_in,
                                          const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 2);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(
      llvm::all_of(layouts_in, [&](const Layout &l) { return l.has_value(); }));
  const Location loc = op.getLoc();
  const VectorLayout &src_layout = *layouts_in[0];
  const VectorLayout &acc_layout = *layouts_in[1];
  const VectorLayout &dst_layout = *layouts_out[0];
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto multi_reduction_op = cast<vector::MultiDimReductionOp>(op);
  const VectorType src_ty = multi_reduction_op.getSourceVectorType();
  auto element_type = src_ty.getElementType();
  int64_t src_rank = src_ty.getRank();
  const auto res_ty = dyn_cast<VectorType>(multi_reduction_op.getDestType());
  if (res_ty == nullptr) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Can only reduce into vectors");
  }
  // Op definition enforces that accumulator type must match result type
  auto acc = cast<TypedValue<VectorType>>(multi_reduction_op.getAcc());
  TPU_ASSERT_OP(layouts_out.front().has_value());

  SmallVector<int64_t> dims(multi_reduction_op.getReductionDims());
  std::sort(dims.begin(), dims.end());

  // Make sure that the accumulator is a splat of the neutral value
  if (acc_layout.offsets() != LayoutOffsets{std::nullopt, std::nullopt}) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Only replicated accumulator supported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> acc_vregs,
      disassemble(builder, acc_layout, acc, ctx.target_shape));
  auto acc_def = dyn_cast_if_present<arith::ConstantOp>(
      acc_vregs.begin()->getDefiningOp());
  if (acc_def == nullptr) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Only constant accumulator supported");
  }
  if (!element_type.isF32() && !element_type.isBF16() &&
      !element_type.isSignlessInteger((32))) {
    return multi_reduction_op.emitOpError(
        "Not implemented: unsupported element type");
  }
  bool is_int = element_type.isSignlessInteger(32);
  const auto acc_def_value = dyn_cast<DenseElementsAttr>(acc_def.getValue());
  if (acc_def_value == nullptr || !acc_def_value.isSplat()) {
    return multi_reduction_op.emitOpError("Expected a splat constant");
  }
  TPU_ASSERT_OP(acc_def_value.getElementType() == element_type);
  Attribute neutral;
  switch (multi_reduction_op.getKind()) {
    case vector::CombiningKind::ADD:
      neutral = builder.getZeroAttr(element_type);
      break;
    case vector::CombiningKind::MAXIMUMF: {
      // TODO(b/322836633): The semantics of maximumf don't match the lowering
      // for older TPU versions because older TPU versions don't respect the
      // -0.0 vs +0.0 ordering.
      neutral = builder.getFloatAttr(
          element_type,
          APFloat::getInf(cast<FloatType>(element_type).getFloatSemantics(),
                          /*Negative=*/true));
    } break;
    case vector::CombiningKind::MINIMUMF: {
      neutral = builder.getFloatAttr(
          element_type,
          APFloat::getInf(cast<FloatType>(element_type).getFloatSemantics(),
                          /*Negative=*/false));
    } break;
    case vector::CombiningKind::MAXSI: {
      neutral = builder.getIntegerAttr(
          element_type,
          APInt::getSignedMinValue(element_type.getIntOrFloatBitWidth()));
    } break;
    case vector::CombiningKind::MINSI: {
      neutral = builder.getIntegerAttr(
          element_type,
          APInt::getSignedMaxValue(element_type.getIntOrFloatBitWidth()));
    } break;
    default:
      return multi_reduction_op.emitOpError(
          "Not implemented: unsupported kind");
  }
  if (auto val = acc_def_value.getSplatValue<Attribute>(); val != neutral) {
    return multi_reduction_op.emitOpError(
               "Not implemented: Only neutral accumulator supported for "
               "float reduction. Expected ")
           << neutral << ", but got " << val;
  }

  std::array<bool, 2> reduces;
  switch (src_layout.implicit_dim()) {
    case VectorLayout::ImplicitDim::kNone:
      reduces = {
          std::find(dims.begin(), dims.end(), src_rank - 2) != dims.end(),
          std::find(dims.begin(), dims.end(), src_rank - 1) != dims.end()};
      break;
    case VectorLayout::ImplicitDim::kSecondMinor:
      reduces = {false, std::find(dims.begin(), dims.end(), src_rank - 1) !=
                            dims.end()};
      break;
    case VectorLayout::ImplicitDim::kMinor:
      reduces = {
          std::find(dims.begin(), dims.end(), src_rank - 1) != dims.end(),
          false};
      break;
  }
  const std::array<bool, 2> allow_replicated = {!reduces[0], !reduces[1]};

  if ((reduces[0] || reduces[1]) &&
      !src_layout.hasNativeTiling(ctx.target_shape)) {
    return multi_reduction_op.emitOpError(
               "Not implemented: Unsupported input layout: ")
           << src_layout;
  }
  if (src_layout.tiling() != dst_layout.tiling()) {
    return multi_reduction_op.emitOpError("Not implemented: Tiling change");
  }
  for (int i = 0; i < 2; ++i) {
    if (reduces[i] && src_layout.offsets()[i] == std::nullopt) {
      return multi_reduction_op.emitOpError(
          "Not implemented: Reductions over replicated axes");
    }
    // Offsets have to be equal, unless we're reducing over that dimension.
    if (src_layout.offsets()[i] != dst_layout.offsets()[i] && !reduces[i]) {
      return multi_reduction_op.emitOpError("Not implemented: Offset change");
    }
  }
  VectorLayout::ImplicitDim dst_implicit_dim;
  if ((reduces[0] && reduces[1]) ||
      (src_layout.implicit_dim() != VectorLayout::ImplicitDim::kNone &&
       (reduces[0] || reduces[1]))) {
    // This is difficult, because we'd like to make both tiling dims implicit,
    // but there is no way to do that in VectorLayout right now.
    // We use an equivalence between VectorLayouts when trailing dims are 1
    // to enable some special cases, but we should generalize this.
    if (*(res_ty.getShape().end() - 1) != 1) {
      return multi_reduction_op.emitOpError(
          "Not implemented: reductions over both trailing dimensions are only "
          "supported when the resulting value has a trailing axis of size 1");
    }
    dst_implicit_dim =
        VectorLayout::ImplicitDim::kSecondMinor;  // Anything works.
  } else if (reduces[0]) {
    TPU_ASSERT_OP(src_layout.implicit_dim() ==
                  VectorLayout::ImplicitDim::kNone);
    dst_implicit_dim = VectorLayout::ImplicitDim::kSecondMinor;
  } else if (reduces[1]) {
    TPU_ASSERT_OP(src_layout.implicit_dim() ==
                  VectorLayout::ImplicitDim::kNone);
    dst_implicit_dim = VectorLayout::ImplicitDim::kMinor;
  } else {
    dst_implicit_dim = src_layout.implicit_dim();
  }
  if (dst_layout.implicit_dim() != dst_implicit_dim) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Unsupported output implicit dimension");
  }

  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_vregs,
      disassemble(builder, src_layout, multi_reduction_op.getSource(),
                  ctx.target_shape));
  xla::Array<Value> dst_vregs(
      dst_layout.tileArrayShape(res_ty.getShape(), ctx.target_shape));
  tpu::ReductionKind tpu_kind;
  switch (multi_reduction_op.getKind()) {
    case vector::CombiningKind::ADD:
      tpu_kind = tpu::ReductionKind::SUM;
      break;
    case vector::CombiningKind::MAXIMUMF:
    case vector::CombiningKind::MAXSI:
      tpu_kind = tpu::ReductionKind::MAX;
      break;
    case vector::CombiningKind::MINIMUMF:
    case vector::CombiningKind::MINSI:
      tpu_kind = tpu::ReductionKind::MIN;
      break;
    default:
      return multi_reduction_op.emitOpError(
          "Not implemented: unsupported reduction kind");
  }
  const ArrayRef<int64_t> src_shape = src_ty.getShape();
  auto all_results_ok = dst_vregs.EachStatus(
      [&](const absl::Span<const int64_t> idx, Value *const dst_vreg) {
        // Extract a subset of source vregs that reduce into this result vreg.
        SmallVector<int64_t> src_slice_start;
        src_slice_start.reserve(src_rank);
        SmallVector<int64_t> src_slice_end;
        src_slice_end.reserve(src_rank);
        for (int64_t i : idx) {
          src_slice_start.push_back(i);
          src_slice_end.push_back(i + 1);
        }
        for (int64_t d : dims) {
          src_slice_start.insert(src_slice_start.begin() + d, 0);
          src_slice_end.insert(src_slice_end.begin() + d, src_vregs.dim(d));
        }
        xla::Array<Value> reduced_vregs =
            src_vregs.Slice(src_slice_start, src_slice_end);
        std::optional<Value> acc_vreg;
        auto reduce_elementwise = [&](Value lhs, Value rhs) -> Value {
          Value result;
          switch (tpu_kind) {
            case tpu::ReductionKind::SUM:
              result =
                  is_int
                      ? builder.create<arith::AddIOp>(loc, lhs, rhs).getResult()
                      : builder.create<arith::AddFOp>(loc, lhs, rhs)
                            .getResult();
              break;
            case tpu::ReductionKind::MAX:
              result = is_int ? builder.create<arith::MaxSIOp>(loc, lhs, rhs)
                                    .getResult()
                              : builder.create<arith::MaximumFOp>(loc, lhs, rhs)
                                    .getResult();
              break;
            case tpu::ReductionKind::MIN:
              result = is_int ? builder.create<arith::MinSIOp>(loc, lhs, rhs)
                                    .getResult()
                              : builder.create<arith::MinimumFOp>(loc, lhs, rhs)
                                    .getResult();
              break;
          }
          return result;
        };
        auto reduction_status = reduced_vregs.EachStatus(
            [&](const absl::Span<const int64_t> red_idx,
                Value *const src_vreg) {
              SmallVector<int64_t> src_idx(red_idx.begin(), red_idx.end());
              for (int i = 0; i < src_idx.size(); ++i) {
                src_idx[i] += src_slice_start[i];
              }
              const std::unique_ptr<VRegDataBounds> data_bounds =
                  src_layout.tileDataBounds(builder.getContext(), src_shape,
                                            src_idx, ctx.target_shape,
                                            allow_replicated);
              if (data_bounds == nullptr) {
                // Op error has already been emitted inside tileDataBounds().
                return absl::UnknownError("Unable to obtain data bounds");
              }
              // TODO(tlongeri): Maybe assemble/disassemble should take
              // TypedValue<VectorType> and we could save casts here and
              // elsewhere
              FailureOr<Value> failure_or_vreg =
                  maskOOB(ctx, builder, cast<TypedValue<VectorType>>(*src_vreg),
                          *data_bounds, neutral);
              if (failed(failure_or_vreg)) {
                op.emitOpError("Failed to mask vreg");
                return absl::UnknownError("");
              }
              Value vreg = failure_or_vreg.value();
              if (!acc_vreg.has_value()) {
                acc_vreg = vreg;
              } else {
                acc_vreg = reduce_elementwise(*acc_vreg, vreg);
              }
              return absl::OkStatus();
            });
        if (!reduction_status.ok()) {
          return reduction_status;
        }
        TPU_ASSERT_OP(acc_vreg.has_value());
        if (reduces[1]) {
          acc_vreg = builder.create<tpu::AllReduceOp>(
              multi_reduction_op->getLoc(), *acc_vreg, 1, tpu_kind);
        }
        if (reduces[0]) {
          // Packed types are compressed along rows, so we need to reduce them
          // within each 32-bit word. There's no performance penalty for doing
          // this in 32-bit precision, so we take advantage of it.
          Type acc_vreg_ty = acc_vreg->getType();
          if (acc_layout.packing() > 1) {
            Type vreg_ty_32 = nullptr;
            if (acc.getType().getElementType().isBF16()) {
              vreg_ty_32 =
                  getNativeVregType(builder.getF32Type(), ctx.target_shape);
            } else {
              multi_reduction_op.emitOpError(
                  "Not implemented: Unsupported reduction dtype");
              return absl::UnknownError("");
            }
            Value acc_vreg_32 = builder.create<tpu::UnpackSubelementsOp>(
                loc, vreg_ty_32, *acc_vreg, 0, tpu::PackFormat::kInterleaved);
            for (int i = 1; i < acc_layout.packing(); ++i) {
              Value acc_vreg_part_32 = builder.create<tpu::UnpackSubelementsOp>(
                  loc, vreg_ty_32, *acc_vreg, i, tpu::PackFormat::kInterleaved);
              acc_vreg_32 = reduce_elementwise(acc_vreg_32, acc_vreg_part_32);
            }
            acc_vreg = acc_vreg_32;
          }
          // At this point acc_vreg is always 32-bit.
          acc_vreg = builder.create<tpu::AllReduceOp>(
              multi_reduction_op->getLoc(), *acc_vreg, 0, tpu_kind);
          // We pack the final result back into the original type.
          if (acc_layout.packing() > 1) {
            SmallVector<int32_t> positions(acc_layout.packing());
            std::iota(positions.begin(), positions.end(),
                      static_cast<int32_t>(0));
            SmallVector<Value> parts(acc_layout.packing(), *acc_vreg);
            acc_vreg = builder.create<tpu::PackSubelementsOp>(
                loc, acc_vreg_ty, parts,
                builder.getDenseI32ArrayAttr(positions),
                tpu::PackFormat::kInterleaved);
          }
        }
        *dst_vreg = *acc_vreg;
        return absl::OkStatus();
      });
  if (!all_results_ok.ok()) {
    return failure();
  }
  multi_reduction_op->replaceAllUsesWith(
      assemble(builder, res_ty, dst_layout, dst_vregs, ctx.target_shape));
  multi_reduction_op->erase();
  return success();
}

LogicalResult vector_shape_cast_rule(RewriteContext &ctx, Operation &op,
                                     const ArrayRef<Layout> layouts_in,
                                     const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  using Tiling = std::array<int64_t, 2>;
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  TPU_ASSERT_EQ_OP(
      layout_in.bitwidth(),
      layout_out.bitwidth());  // This should be guaranteed through MLIR
                               // verifier plus our layoutIsValidForValue check
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto shape_cast_op = cast<vector::ShapeCastOp>(op);
  const VectorType src_ty = shape_cast_op.getSourceVectorType();
  const ArrayRef<int64_t> src_shape = src_ty.getShape();
  const VectorType dst_ty = shape_cast_op.getResultVectorType();
  const ArrayRef<int64_t> dst_shape = dst_ty.getShape();
  bool no_op = false;
  const std::array<int64_t, 2> src_tiled_dims =
      layout_in.getImplicitTiledDims(src_shape, 1);
  const std::array<int64_t, 2> dst_tiled_dims =
      layout_out.getImplicitTiledDims(dst_shape, 1);
  const std::array<int64_t, 2> src_vreg_slice =
      layout_in.vregSlice(ctx.target_shape);
  const std::array<int64_t, 2> dst_vreg_slice =
      layout_out.vregSlice(ctx.target_shape);
  if (layout_in.tiling() == layout_out.tiling() &&
      layout_in.offsets() == layout_out.offsets() &&
      src_tiled_dims == dst_tiled_dims) {
    no_op = true;
  } else if (  // Fold or unfold sublane dim, but keeping a whole number of
               // vregs.
      layout_in.offsets()[0] == 0 &&
      layout_in.offsets() == layout_out.offsets() &&
      layout_in.tiling() == layout_out.tiling() &&
      dst_tiled_dims[1] == src_tiled_dims[1] &&
      dst_tiled_dims[0] % dst_vreg_slice[0] == 0 &&
      src_tiled_dims[0] % src_vreg_slice[0] == 0) {
    no_op = true;
  } else if (layout_in.offsets() == layout_out.offsets() &&
             layout_in.offsets() == LayoutOffsets{0, 0} &&
             layout_in.tiling()[0] == 1 &&
             layout_out.hasNativeTiling(ctx.target_shape) &&
             dst_tiled_dims[1] == dst_vreg_slice[1] &&
             dst_tiled_dims[0] % dst_vreg_slice[0] == 0 &&
             src_tiled_dims[1] % src_vreg_slice[1] == 0) {
    // Shapecast (..., m * 128 * packing) -> (..., 128).
    no_op = true;
  } else if (layout_in.offsets() == LayoutOffsets{0, 0} &&
             layout_out.offsets() == LayoutOffsets{0, 0} &&
             layout_in.hasNativeTiling(ctx.target_shape) &&
             layout_out.tiling()[0] == 1 &&
             src_tiled_dims[1] == src_vreg_slice[1] &&
             src_tiled_dims[0] % src_vreg_slice[0] == 0 &&
             dst_tiled_dims[1] % dst_vreg_slice[1] == 0) {
    // Shapecast (..., 128) -> (..., m * 128 * packing).
    no_op = true;
  } else if (layout_in.offsets() == LayoutOffsets{0, 0} &&
             layout_out.offsets() == LayoutOffsets{0, 0} &&
             layout_in.tiling()[0] == 1 && layout_out.tiling()[0] == 1 &&
             src_vreg_slice[1] == dst_vreg_slice[1] &&
             src_tiled_dims[1] % src_vreg_slice[1] == 0 &&
             dst_tiled_dims[1] % dst_vreg_slice[1] == 0) {
    no_op = true;
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_vregs,
      disassemble(builder, layout_in, shape_cast_op.getSource(),
                  ctx.target_shape, /*use_implicit_shape=*/true));
  auto getDstVregs = [&]() -> FailureOr<xla::Array<Value>> {
    if (no_op) {
      xla::Array<Value> dst_vregs_local = src_vregs;
      dst_vregs_local.Reshape(
          layout_out.tileArrayImplicitShape(dst_shape, ctx.target_shape));
      return dst_vregs_local;
    } else if (dst_tiled_dims == std::array<int64_t, 2>{src_tiled_dims[1], 1} &&
               layout_in.bitwidth() == 32 &&
               layout_in.hasNativeTiling(ctx.target_shape) &&
               layout_in.tiling() == layout_out.tiling() &&
               (!layout_in.offsets()[1].has_value() ||
                *layout_in.offsets()[1] % ctx.target_shape[0] ==
                    layout_out.offsets()[0] ||
                *layout_in.offsets()[1] + src_tiled_dims[1] <=
                    ctx.target_shape[1])) {
      FAILUREOR_ASSIGN_OR_RETURN(
          xla::Array<Value> dst_vregs_local,
          insertImplicitMinorDimension(ctx, builder, op.getLoc(), src_vregs,
                                       layout_in.implicitShape(src_shape),
                                       layout_in, layout_out.offsets()));
      // Now, reshape the major axes of the vreg array.
      dst_vregs_local.Reshape(
          layout_out.tileArrayImplicitShape(dst_shape, ctx.target_shape));
      return dst_vregs_local;
    } else {
      return shape_cast_op.emitOpError(
                 "Not implemented: Unsupported vector.shape_cast: ")
             << *shape_cast_op;
    }
  };
  FAILUREOR_ASSIGN_OR_RETURN(const xla::Array<Value> dst_vregs, getDstVregs());
  shape_cast_op->replaceAllUsesWith(assemble(builder, dst_ty, layout_out,
                                             dst_vregs, ctx.target_shape,
                                             /*use_implicit_shape=*/true));
  shape_cast_op->erase();
  return success();
}

template <typename Op>
LogicalResult vector_store_impl(RewriteContext &ctx, Op store_op,
                                const VectorLayout &to_store_layout,
                                TypedValue<VectorType> store_mask = nullptr) {
  Operation &op = *(store_op.getOperation());
  MLIRContext *const mlir_ctx = store_op.getContext();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  const VectorType ty = store_op.getValueToStore().getType();
  const auto memref_ty = getMemRefType(store_op.getBase());
  if (!ty.getRank()) {
    return op.emitOpError("Not implemented: scalar stores to vmem");
  }
  const bool is_1d = ty.getRank() == 1;
  VectorLayout::ImplicitDim expected_dim =
      is_1d ? VectorLayout::ImplicitDim::kSecondMinor
            : VectorLayout::ImplicitDim::kNone;
  if (to_store_layout.implicit_dim() != expected_dim) {
    return op.emitOpError("Not implemented: unsupported layout");
  }
  using Tiling = std::array<int64_t, 2>;
  FAILUREOR_ASSIGN_OR_RETURN(
      const Tiling memref_tiling,
      getMemRefTiling(store_op.getBase(), ctx.target_shape));
  if (memref_tiling != to_store_layout.tiling()) {
    if (memref_tiling[0] == 1 && to_store_layout.tiling()[0] == 1 &&
        memref_tiling[1] % to_store_layout.tiling()[1] == 0) {
      // In this case, it is valid to have to_store tiling (1, 128 * packing)
      // when storing to a 1D memref.
    } else if (to_store_layout.bitwidth() == 32 &&
               to_store_layout.tiling() ==
                   std::array<int64_t, 2>{1, ctx.target_shape[1]}) {
      // In this case, it is valid to have to_store tiling (1,
      // TARGET_SHAPE.lanes) because we strided-store one row to each tile of
      // the memref. This can save us a bunch of stores!
      // TODO(b/295393167): need to support strided store for bitwidth < 32.
    } else if (to_store_layout.bitwidth() == 32 &&
               // We accept padding in the minormost dim, because
               // apply_vector_layout will properly mask stores。
               canReinterpretToUntiledMemref(
                   store_op.getBase(), ctx.target_shape,
                   /*allow_minormost_padding=*/true)) {
      // In this case, if the memref can be reinterpreted to untiled, it is
      // valid to use any tiling for to_store. But using native tiling can save
      // us a bunch of stores!
    } else {
      return op.emitOpError(
          "Not implemented: dismatch in memref tiling and vector tiling in "
          "store");
    }
  }

  bool can_support_unaligned_dynamic_index = false;
  bool must_support_unaligned_dynamic_index = false;
  if (store_op.getIndices().size() > 1) {
    auto second_minor_idx = store_op.getIndices().take_back(2)[0];
    if (failed(getIntConst(second_minor_idx, /*silent=*/true)) &&
        !isGuaranteedDivisible(second_minor_idx, memref_tiling[0])) {
      must_support_unaligned_dynamic_index = true;
    }
  }
  int64_t sublane_stride = 1;
  // Handle special patterns that allow us to support more flexible loads.
  if (to_store_layout.bitwidth() == 32 &&
      to_store_layout.tiling() == Tiling{1, ctx.target_shape[1]}) {
    // Storing a single row on the 2nd minor dim from the (1, 128) layout. We
    // can use sublane striding to perform the relayout as part of the store.
    // The stride of store should be the number of sublanes in memref tile when
    // store a single sublane.
    sublane_stride = memref_tiling[0];
    can_support_unaligned_dynamic_index = true;
  } else {
    // Otherwise, if the memref has a short last dimension and is contiguous
    // all the tiled layouts become equivalent, so we can handle unaligned
    // dynamic indices without any special case.
    auto mem_layout = dyn_cast<TiledLayoutAttr>(memref_ty.getLayout());
    if (!mem_layout) {
      return op.emitOpError("Expected a tiled memref");
    }
    auto tile_strides = mem_layout.getTileStrides();
    if (memref_ty.getShape().back() == ctx.target_shape[1] &&
        tile_strides.take_back(2) == ArrayRef<int64_t>{1, 1}) {
      can_support_unaligned_dynamic_index = true;
    }
  }

  auto add_idx = [&](const Value &v, int64_t d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return IdxConst(cst.value() + d, builder, op.getLoc());
    }
    return builder.create<arith::AddIOp>(v, IdxConst(d, builder, op.getLoc()));
  };

  int tiled_dims = is_1d ? 1 : 2;
  Value base_addr = store_op.getBase();
  SmallVector<Value, 4> base_indices = store_op.getIndices();

  if (must_support_unaligned_dynamic_index) {
    if (!can_support_unaligned_dynamic_index) {
      return op.emitOpError(
          "Not implemented: dynamic store with unaligned indices");
    }
  } else {
    // Convert dynamic store to dynamic slice + static store. This saves us a
    // bunch of scalar core work.
    auto slice_result = sliceRef(
        builder, store_op.getBase(), ty.getShape(), store_op.getIndices(),
        ArrayRef<int64_t>(memref_tiling).take_back(tiled_dims));
    if (failed(slice_result)) {
      return failure();
    }
    base_addr = slice_result->first;
    CHECK_EQ(slice_result->second.size(), base_indices.size());
    for (int i = 0; i < base_indices.size(); ++i) {
      base_indices[i] = IdxConst(slice_result->second[i], builder, op.getLoc());
    }
  }

  // TODO(jevinjiang): ideally we should update the base addr and use static
  // indices even for the cases that can skip alignment check. This can save
  // us a bunch of scalar core work.
  auto tile_base_idxs = ArrayRef<Value>(base_indices).take_back(tiled_dims);
  auto batch_base_idxs = ArrayRef<Value>(base_indices).drop_back(tiled_dims);

  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> tiles,
      disassemble(builder, to_store_layout, store_op.getValueToStore(),
                  ctx.target_shape, /*use_implicit_shape=*/true));
  std::optional<xla::Array<Value>> tile_masks;
  if (store_mask) {
    FAILUREOR_ASSIGN_OR_RETURN(
        tile_masks, disassemble(builder, to_store_layout, store_mask,
                                ctx.target_shape, /*use_implicit_shape=*/true));
    TPU_ASSERT_EQ_OP(tile_masks->dimensions(), tiles.dimensions());
  }
  const int64_t ndims = ty.getRank();
  const auto base_s = is_1d ? nullptr : tile_base_idxs.front();
  const auto base_l = tile_base_idxs.back();
  const LayoutOffset sublane_offset = to_store_layout.offsets()[0];
  const LayoutOffset lane_offset = to_store_layout.offsets()[1];
  if (!sublane_offset.has_value() || !lane_offset.has_value()) {
    return store_op.emitOpError(
        "Not implemented: Replicated layout disallowed in vector store");
  }
  const SmallVector<int64_t> stored_shape =
      to_store_layout.implicitShape(ty.getShape());
  const std::array<int64_t, 2> vreg_slice =
      to_store_layout.vregSlice(ctx.target_shape);
  const absl::Status status =
      tiles.EachStatus([&](const absl::Span<const int64_t> idx,
                           const Value tile) -> absl::Status {
        const auto tile_mask = store_mask ? (*tile_masks)(idx) : nullptr;
        const std::unique_ptr<VRegDataBounds> bounds =
            to_store_layout.tileDataBounds(mlir_ctx, stored_shape,
                                           toArrayRef(idx), ctx.target_shape);
        const int64_t sidx = *(idx.end() - 2);
        const int64_t lidx = *(idx.end() - 1);
        SmallVector<Value> indices(ndims);
        for (int64_t i = 0; i < batch_base_idxs.size(); ++i) {
          indices[i] = add_idx(batch_base_idxs[i], idx[i]);
        }
        if (!is_1d) {
          *(indices.end() - 2) =
              add_idx(base_s, sidx * vreg_slice[0] - *sublane_offset);
        }
        *(indices.end() - 1) =
            add_idx(base_l, lidx * vreg_slice[1] - *lane_offset);
        const DenseBoolArrayAttr sublane_mask =
            bounds->getSublaneMask(store_op->getContext(), ctx.target_shape);
        const bool masks_subelements =
            bounds->maskVariesAlong(Direction::kSubelements, ctx.target_shape);
        if (bounds->maskVariesAlong(Direction::kLanes, ctx.target_shape) ||
            masks_subelements) {
          auto failure_or_mask =
              bounds->getVectorMask(builder, store_op.getLoc(),
                                    ctx.hardware_generation, ctx.target_shape);
          if (failed(failure_or_mask)) {
            return absl::UnimplementedError("Failed to get vector mask");
          }
          TypedValue<VectorType> mask = failure_or_mask.value();
          // Vmem stores don't support masking below 32-bit granularity, so we
          // need to load and blend explicitly if needed.
          if (masks_subelements) {
            auto data = builder.create<tpu::LoadOp>(tile.getType(), base_addr,
                                                    indices, sublane_mask,
                                                    /*sublane_stride=*/nullptr);
            const bool mask_is_a_bitmask =
                cast<IntegerType>(mask.getType().getElementType()).getWidth() ==
                32;
            Value updated;
            if (mask_is_a_bitmask) {
              auto ones = builder.create<arith::ConstantOp>(
                  mask.getType(),
                  DenseElementsAttr::get(
                      mask.getType(),
                      builder.getIntegerAttr(builder.getI32Type(),
                                             APInt(32, 0xFFFFFFFF))));
              auto masked_tile = builder.create<arith::AndIOp>(
                  store_op.getLoc(), mask,
                  builder.create<tpu::BitcastVregOp>(mask.getType(), tile));
              auto mask_neg = builder.create<arith::XOrIOp>(ones, mask);
              auto masked_data = builder.create<arith::AndIOp>(
                  mask_neg,
                  builder.create<tpu::BitcastVregOp>(mask.getType(), data));
              updated = builder.create<tpu::BitcastVregOp>(
                  tile.getType(),
                  builder.create<arith::OrIOp>(masked_data, masked_tile));
            } else {
              updated = builder.create<arith::SelectOp>(mask, tile, data);
            }
            builder.create<tpu::StoreOp>(
                updated, base_addr, indices, sublane_mask, tile_mask,
                /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
          } else {
            builder.create<tpu::StoreOp>(
                tile, base_addr, indices, sublane_mask,
                tile_mask
                    ? builder.create<arith::AndIOp>(mask, tile_mask).getResult()
                    : mask,
                /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
          }
        } else {
          builder.create<tpu::StoreOp>(
              tile, base_addr, indices, sublane_mask, tile_mask,
              /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
        }
        return absl::OkStatus();
      });
  if (!status.ok()) {
    return failure();
  }
  store_op->erase();
  return success();
}

LogicalResult vector_store_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  auto store_op = cast<vector::StoreOp>(op);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(llvm::none_of(layouts_in.drop_front(),
                              [&](const Layout &l) { return l.has_value(); }));
  return vector_store_impl(ctx, store_op, *layouts_in.front());
}

LogicalResult tpu_vector_store_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  auto store_op = cast<tpu::VectorStoreOp>(op);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  auto other_layouts_in = layouts_in.drop_front();
  if (store_op.getMask()) {
    TPU_ASSERT_EQ_OP(layouts_in.front(), layouts_in.back());
    other_layouts_in = other_layouts_in.drop_back();
  }
  TPU_ASSERT_OP(llvm::none_of(other_layouts_in,
                              [&](const Layout &l) { return l.has_value(); }));
  return vector_store_impl(ctx, store_op, *layouts_in.front(),
                           store_op.getMask());
}

LogicalResult vector_transpose_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_in != layout_out) {
    return op.emitOpError("Not implemented: Unsupported 2D layouts");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto transpose_op = cast<vector::TransposeOp>(op);
  VectorType src_ty = transpose_op.getSourceVectorType();
  VectorType dst_ty = transpose_op.getResultVectorType();
  const int64_t rank = src_ty.getRank();
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_vregs,
      disassemble(builder, layout_in, transpose_op.getVector(),
                  ctx.target_shape));
  ArrayRef<int64_t> permutation = transpose_op.getPermutation();
  const auto tile_perm = permutation.take_back(2);
  if (tile_perm != ArrayRef<int64_t>{rank - 2, rank - 1} &&
      tile_perm != ArrayRef<int64_t>{rank - 1, rank - 2}) {
    return transpose_op->emitOpError(
        "Not implemented: Unsupported permutation");
  }
  {
    SmallVector<int64_t> p(permutation);
    p[rank - 2] = rank - 2;
    p[rank - 1] = rank - 1;
    src_vregs.TransposeDimensions(p);
  }
  if (tile_perm == ArrayRef<int64_t>{rank - 2, rank - 1}) {
    transpose_op->replaceAllUsesWith(
        assemble(builder, dst_ty, layout_out, src_vregs, ctx.target_shape));
    transpose_op.erase();
    return success();
  }
  if (layout_in.offsets() != LayoutOffsets{0, 0} ||
      !layout_in.hasNativeTiling(ctx.target_shape)) {
    return transpose_op->emitOpError(
        "Not implemented: Non-native or offset layout unsupported");
  }
  const int64_t transpose_unit_size = ctx.target_shape[1];
  if (ctx.hardware_generation < 4 && layout_in.bitwidth() != 32) {
    return transpose_op->emitOpError(
        "Not implemented: TPUs before v4 only support 32-bit transposes");
  }
  xla::Array<Value> dst_vregs(
      layout_out.tileArrayShape(dst_ty.getShape(), ctx.target_shape));
  const int packing = layout_in.packing();
  // Note that we checked for native tiling above.
  const int64_t vregs_per_tile = transpose_unit_size / layout_in.tiling()[0];
  const SmallVector<int64_t> minor_perm{1, 0};
  const auto tile_ty = VectorType::get(
      {transpose_unit_size, transpose_unit_size}, src_ty.getElementType());
  const auto batch_tile_ty_in =
      VectorType::get({transpose_unit_size, transpose_unit_size * packing},
                      src_ty.getElementType());
  const auto batch_tile_ty_out =
      VectorType::get({transpose_unit_size * packing, transpose_unit_size},
                      src_ty.getElementType());
  // For packed types, we can increase the XLU throughput by batching together
  // multiple tiles. At the moment we always batch along columns, with the
  // reasoning being that if all the tiles are fed into the MXU, then it's
  // better if we end up with results that contribute to the same contraction.
  const bool can_batch =
      layout_in.bitwidth() == 16 && ctx.hardware_generation < 6;
  auto doTranspose = [&](const ArrayRef<int64_t> batch_idx,
                         const int64_t src_row, const int64_t src_col,
                         const int64_t src_col_end, const VectorType tile_ty_in,
                         const VectorType tile_ty_out) {
    SmallVector<int64_t> src_slice_starts;
    src_slice_starts.reserve(rank);
    src_slice_starts.append(batch_idx.begin(), batch_idx.end());
    src_slice_starts.append({src_row * vregs_per_tile, src_col});
    SmallVector<int64_t> src_slice_ends;
    src_slice_ends.reserve(rank);
    auto incremented_batch_idx =
        map_range(batch_idx, [](int64_t i) { return i + 1; });
    src_slice_ends.append(incremented_batch_idx.begin(),
                          incremented_batch_idx.end());
    src_slice_ends.append({(src_row + 1) * vregs_per_tile, src_col_end});
    xla::Array<Value> src_tile_vregs = src_vregs.Slice(
        src_slice_starts, src_slice_ends,
        builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getZeroAttr(src_vregs.begin()->getType())));
    // Drop leading singleton (batch) dimensions to have a shape that conforms
    // with the vreg array shape specified by layout_in, as expected by assemble
    src_tile_vregs.Reshape(
        ArrayRef<int64_t>{vregs_per_tile, src_col_end - src_col});
    const Value src_tile = assemble(builder, tile_ty_in, layout_in,
                                    src_tile_vregs, ctx.target_shape);
    auto new_transpose_op =
        builder.create<vector::TransposeOp>(tile_ty_out, src_tile, minor_perm);
    new_transpose_op->setAttr("out_layout",
                              builder.getAttr<VectorLayoutAttr>(layout_out));
    auto unroll_vectors_op = builder.create<tpu::UnrollVectorsOp>(
        llvm::map_to_vector(src_tile_vregs,
                            [](Value v) { return v.getType(); }),
        new_transpose_op);
    SmallVector<int64_t> dst_slice_starts;
    dst_slice_starts.reserve(rank);
    dst_slice_starts.append(batch_idx.begin(), batch_idx.end());
    dst_slice_starts.append({src_col * vregs_per_tile, src_row});
    SmallVector<int64_t> dst_slice_ends;
    dst_slice_ends.reserve(rank);
    dst_slice_ends.append(incremented_batch_idx.begin(),
                          incremented_batch_idx.end());
    dst_slice_ends.append({src_col_end * vregs_per_tile, src_row + 1});
    updateSliceFromRange(dst_vregs, unroll_vectors_op.getResults(),
                         dst_slice_starts, dst_slice_ends);
  };
  const int num_batch_dims = rank - 2;
  const ArrayRef<int64_t> batch_sizes =
      dst_ty.getShape().take_front(num_batch_dims);
  SmallVector<int64_t> batch_idx(num_batch_dims);
  const int64_t tile_rows =
      xla::CeilOfRatio(*(src_ty.getShape().end() - 2), transpose_unit_size);
  const int64_t num_col_tiles =
      xla::CeilOfRatio(*(src_ty.getShape().end() - 1), transpose_unit_size);
  do {
    for (int64_t src_row = 0; src_row < tile_rows; ++src_row) {
      if (can_batch) {
        const int64_t num_batch_tiles = num_col_tiles / 2;
        for (int64_t src_col = 0; src_col < num_batch_tiles; ++src_col) {
          doTranspose(batch_idx, src_row, src_col * 2, (src_col + 1) * 2,
                      batch_tile_ty_in, batch_tile_ty_out);
        }
        if (num_col_tiles % 2 == 1) {
          doTranspose(batch_idx, src_row, num_col_tiles - 1, num_col_tiles,
                      tile_ty, tile_ty);
        }
      } else {
        for (int64_t src_col = 0; src_col < num_col_tiles; ++src_col) {
          doTranspose(batch_idx, src_row, src_col, src_col + 1, tile_ty,
                      tile_ty);
        }
      }
    }
  } while (incrementIndex(batch_idx, batch_sizes));
  for (const Value v : dst_vregs) {
    TPU_ASSERT_OP(v != nullptr);
  }
  transpose_op->replaceAllUsesWith(
      assemble(builder, dst_ty, layout_out, dst_vregs, ctx.target_shape));
  transpose_op->erase();
  return success();
}

LogicalResult prng_random_bits_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());

  const VectorLayout &layout_out = *layouts_out.front();
  tpu::PRNGRandomBitsOp rng_op = cast<tpu::PRNGRandomBitsOp>(op);
  if (layout_out != VectorLayout(32, {0, 0}, ctx.target_shape,
                                   VectorLayout::ImplicitDim::kNone)) {
    return op.emitOpError(
        "Unsupported output layout for ") << rng_op->getName();
  }
  OpBuilder builder(op.getContext());
  builder.setInsertionPointAfter(&op);

  VectorType vty = rng_op.getResult().getType();
  TPU_ASSERT_OP(vty.getElementType().isInteger());
  // Only 32-bit output supported currently.
  TPU_ASSERT_OP(vty.getElementType().getIntOrFloatBitWidth() == 32);
  xla::Array<Value> tiles(
      layout_out.tileArrayShape(vty.getShape(), ctx.target_shape));
  VectorType tile_ty = VectorType::get(ctx.target_shape, vty.getElementType());
  tiles.Each([&](absl::Span<const int64_t> tile_idxs, Value * v) {
    *v = builder.create<tpu::PRNGRandomBitsOp>(op.getLoc(), tile_ty);
  });
  const RollVectorsOp roll_vectors_op =
      assemble(builder, vty, layout_out, tiles, ctx.target_shape);
  rng_op->replaceUsesWithIf(roll_vectors_op, [&](OpOperand &operand) {
    return operand.getOwner() != roll_vectors_op;
  });
  rng_op->erase();
  return success();
}

const llvm::StringMap<rule_type> &rules() {
  static const llvm::StringMap<rule_type> *rules = [] {
    static auto rules = new llvm::StringMap<rule_type>{
        {arith::ConstantOp::getOperationName(), arith_constant_rule},
        {arith::ExtFOp::getOperationName(), arith_extf_rule},
        {arith::ExtSIOp::getOperationName(), arith_extsi_rule},
        {arith::ExtUIOp::getOperationName(), arith_extui_rule},
        {arith::TruncFOp::getOperationName(), arith_truncf_rule},
        {arith::TruncIOp::getOperationName(), arith_trunci_rule},
        {func::ReturnOp::getOperationName(), func_return_rule},
        {scf::ForOp::getOperationName(), scf_for_rule},
        {scf::WhileOp::getOperationName(), scf_while_rule},
        {scf::ConditionOp::getOperationName(), scf_condition_rule},
        {scf::IfOp::getOperationName(), scf_if_rule},
        {scf::YieldOp::getOperationName(), yield_rule},
        {tpu::YieldOp::getOperationName(), yield_rule},
        {tpu::RotateOp::getOperationName(), tpu_rotate_rule},
        {tpu::DynamicRotateOp::getOperationName(), tpu_dynamic_rotate_rule},
        {tpu::ConcatenateOp::getOperationName(), tpu_concatenate_rule},
        {tpu::IotaOp::getOperationName(), tpu_iota_rule},
        {tpu::GatherOp::getOperationName(), tpu_gather_rule},
        {tpu::LoadOp::getOperationName(), tpu_load_rule},
        {tpu::StoreOp::getOperationName(), tpu_store_rule},
        {tpu::StridedLoadOp::getOperationName(), tpu_strided_load_rule},
        {tpu::StridedStoreOp::getOperationName(), tpu_strided_store_rule},
        {tpu::VectorStoreOp::getOperationName(), tpu_vector_store_rule},
        {tpu::MatmulOp::getOperationName(), tpu_matmul_rule},
        {tpu::RegionOp::getOperationName(), tpu_region_rule},
        {tpu::BitcastOp::getOperationName(), tpu_bitcast_rule},
        {tpu::TraceOp::getOperationName(), tpu_trace_rule},
        {tpu::AssumeLayoutOp::getOperationName(), tpu_assume_layout_rule},
        {tpu::PRNGRandomBitsOp::getOperationName(), prng_random_bits_rule},
        {vector::BroadcastOp::getOperationName(), vector_broadcast_rule},
        {vector::ExtractOp::getOperationName(), vector_extract_rule},
        {vector::LoadOp::getOperationName(), vector_load_rule},
        {vector::MultiDimReductionOp::getOperationName(),
         vector_multi_reduction_rule},
        {vector::ExtractStridedSliceOp::getOperationName(),
         vector_extract_strided_slice_rule},
        {vector::ShapeCastOp::getOperationName(), vector_shape_cast_rule},
        {vector::StoreOp::getOperationName(), vector_store_rule},
        {vector::TransposeOp::getOperationName(), vector_transpose_rule}};

    llvm::StringMap<rule_type> extended_rules = mlir::tpu::extensions::rules();
    for (auto &entry : extended_rules) {
      rules->insert(&entry);
    }
    return rules;
  }();
  return *rules;
}

// Determines whether we should handle bank conflict for the given stride and
// max_sublane_offset.
//
// See `handleBankConflict` for how this is done.
bool shouldHandleBankConflict(const ApplyVectorLayoutContext &ctx,
                              int32_t stride, int max_sublane_offset) {
  return ctx.hardware_generation >= 4 && ctx.vmem_banks > 0 &&
         ctx.vmem_banks < stride * ctx.target_shape[0] &&
         ctx.max_shuffle_sublane_offset > 0 &&
         ctx.max_shuffle_sublane_offset >= max_sublane_offset;
}

// Handles load/store bank conflict by adding one extra sublane to stride and
// adjusting sublane offsets accordingly.
//
// For example, when store stride is 4 and load sublane offsets are
// [0, 1, 2, 3, 4, 5, 6, 7], the store bank conflict can be avoided by changing
// stride to 5 and sublane offsets to [0, 1, 2, 3, 5, 6, 7, 8].
void handleBankConflict(int32_t &stride, absl::Span<int> sublane_offsets) {
  // Add one extra sublane to stride to avoid bank conflict.
  for (int i = 0; i < sublane_offsets.size(); ++i) {
    // Adjust sublane offsets to match the stride.
    sublane_offsets[i] += i / stride;
  }
  ++stride;
}

}  // namespace

RollVectorsOp assemble(OpBuilder &builder, VectorType vty,
                       const VectorLayout &layout,
                       const xla::Array<Value> &vals,
                       const std::array<int64_t, 2> target_shape,
                       const bool use_implicit_shape) {
  // TODO(tlongeri): Maybe just add a parameter to tileArrayShape instead of
  // having `tileArrayShape` and `tileArrayImplicitShape`.
  SmallVector<int64_t> vreg_array_shape =
      layout.tileArrayImplicitShape(vty.getShape(), target_shape);
  if (!use_implicit_shape) {
    layout.eraseImplicit(vreg_array_shape);
  }
  CHECK(vals.dimensions() == vreg_array_shape);
  CHECK_GT(vals.num_elements(), 0);
  Location loc = vals.begin()->getLoc();
  auto op =
      builder.create<RollVectorsOp>(loc, vty, XlaArrayToFlatArrayRef(vals));
  op->setAttr("out_layout", builder.getAttr<ArrayAttr>(ArrayRef<Attribute>{
                                builder.getAttr<VectorLayoutAttr>(layout)}));
  return op;
}

// Disassemble an MLIR vector into an ndarray of native vectors.
//
// Args:
//   layout: The layout of val. Used to determine the unrolling into
//     native-shaped vectors.
//   val: Value to disassemble. Must be of type VectorType.
//
// Returns:
//   An ndarray of MLIR values representing the tiling of val given by layout.
FailureOr<xla::Array<Value>> disassemble(
    OpBuilder &builder, const VectorLayout &layout,
    const TypedValue<VectorType> val, const std::array<int64_t, 2> target_shape,
    const bool use_implicit_shape) {  // TODO(tlongeri): Remove default
  const auto vty = val.getType();
  const auto op_result = dyn_cast<OpResult>(val);
  if (op_result == nullptr) {
    return failure();
  }
  Operation *const op = op_result.getOwner();
  const unsigned res_idx = op_result.getResultNumber();
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                             getOutLayouts(*op, target_shape));
  const Layout def_layout = def_layouts[res_idx];
  TPU_ASSERT_LOC(val.getLoc(), def_layout.has_value());
  TPU_ASSERT_LOC(val.getLoc(),
                 def_layout->generalizes(layout, vty.getShape(), target_shape));
  auto layout_product =
      xla::Product(layout.tileArrayShape(vty.getShape(), target_shape));
  auto def_layout_product =
      xla::Product(def_layout->tileArrayShape(vty.getShape(), target_shape));
  TPU_ASSERT_LOC(val.getLoc(), layout_product == def_layout_product);
  // TODO(tlongeri): Maybe just add a parameter to tileArrayShape instead of
  // having `tileArrayShape` and `tileArrayImplicitShape`.
  SmallVector<int64_t> layout_shape =
      layout.tileArrayImplicitShape(vty.getShape(), target_shape);
  if (!use_implicit_shape) {
    layout.eraseImplicit(layout_shape);
  }
  if (auto roll_vectors_op = dyn_cast<RollVectorsOp>(op)) {
    return XlaArrayFromShapeAndValues<Value>(layout_shape,
                                             roll_vectors_op->getOperands());
  }
  return op->emitOpError("Not implemented: ") << val;
}

// Assembles a destination tile using partial data from rotated vregs using a
// divide-and-conquer strategy.
//
// Arguments:
//   rotated_row_vregs: A row of rotated vregs, from which destination tile(s)
//     is/are to be selected to assemble a new vreg.
//   src_layout: The source layout.
//   start_src_col: The first rotated vreg in the row of rotated vregs to
//     process.
//   end_src_col: The last rotated vreg in the row of rotated vreg to process.
//   first_dst_tile_sublane_offset: Sublane offset where the first dst tile to
//   be
//     selected starts.
//   dst_layout: Destination layout, based on which retiling is being performed.
//   hw_generation: The generation of a target hardware.
//
// Returns:
//   A new vreg assembled from dst tiles stored in given rotated vregs.
Value selectTilesFromRotatedRowVregs(
    OpBuilder &builder, const ArrayRef<Value> &rotated_row_vregs,
    const int64_t start_src_col, const int64_t end_src_col,
    const int64_t first_dst_tile_sublane_offset, const VectorLayout &dst_layout,
    const std::array<int64_t, 2> target_shape) {
  CHECK_LE(start_src_col, end_src_col);
  CHECK_LE(start_src_col, end_src_col);
  if (start_src_col == end_src_col) {
    return rotated_row_vregs[start_src_col];
  }
  const int64_t mid_src_col = start_src_col + (end_src_col - start_src_col) / 2;

  Value left_partial_vreg = selectTilesFromRotatedRowVregs(
      builder, rotated_row_vregs, start_src_col, mid_src_col,
      first_dst_tile_sublane_offset, dst_layout, target_shape);
  Location loc = left_partial_vreg.getLoc();

  const int64_t left_tiles_count = mid_src_col - start_src_col + 1;
  const int64_t right_first_dst_tile_sublane_offset =
      (first_dst_tile_sublane_offset +
       left_tiles_count * dst_layout.sublanesPerTile(target_shape)) %
      target_shape[0];

  Value right_partial_vreg = selectTilesFromRotatedRowVregs(
      builder, rotated_row_vregs, mid_src_col + 1, end_src_col,
      right_first_dst_tile_sublane_offset, dst_layout, target_shape);

  const IntegerType i1 = builder.getI1Type();
  // We never need to select partial sublanes, even for packed data.
  const auto mask_vreg_ty = VectorType::get(target_shape, i1);
  auto i32_vreg = VectorType::get(target_shape, builder.getI32Type());
  auto select_32bit = [&](Value sublane_mask, Value left, Value right) {
    // Always do the selects on 32-bit granularity for maximum HW compatibility.
    Type vreg_ty = left.getType();
    if (dst_layout.packing() != 1) {
      left = builder.create<tpu::BitcastVregOp>(loc, i32_vreg, left);
      right = builder.create<tpu::BitcastVregOp>(loc, i32_vreg, right);
    }
    Value result =
        builder.create<arith::SelectOp>(loc, sublane_mask, left, right);
    if (dst_layout.packing() != 1) {
      result = builder.create<tpu::BitcastVregOp>(loc, vreg_ty, result);
    }
    return result;
  };

  auto boundIdxConst = std::bind(IdxConst, std::placeholders::_1, builder,
                                 left_partial_vreg.getLoc());
  if (first_dst_tile_sublane_offset < right_first_dst_tile_sublane_offset) {
    // The useful data sublanes in left vregs do not wrap around in vreg.
    // For e.g. consider (2,128) destination tiling and we are trying to merge
    // two vregs as follows:
    //
    //   vreg 0:        vreg 1:
    //   x x x x x     dst_tile_2
    //   x x x x x     dst_tile_3
    //   dst_tile_4    x x x x x
    //   dst_tile_5    x x x x x
    //   dst_tile_6    x x x x x
    //   dst_tile_7    x x x x x
    //   x x x x x     dst_tile_0
    //   x x x x x     dst_tile_1
    //
    // In the above case, the data we want to select from vreg 1 wraps around,
    // whereas vreg 0 useful data is contiguous. It is easier to create '1' mask
    // for vreg 0.
    auto sublanes_mask = builder.create<tpu::CreateMaskOp>(
        left_partial_vreg.getLoc(), mask_vreg_ty,
        ArrayRef<Value>{boundIdxConst(first_dst_tile_sublane_offset),
                        boundIdxConst(0)},
        ArrayRef<Value>{boundIdxConst(right_first_dst_tile_sublane_offset),
                        boundIdxConst(target_shape[1])});
    return select_32bit(sublanes_mask, left_partial_vreg, right_partial_vreg);
  }

  auto sublanes_mask = builder.create<tpu::CreateMaskOp>(
      left_partial_vreg.getLoc(), mask_vreg_ty,
      ArrayRef<Value>{boundIdxConst(right_first_dst_tile_sublane_offset),
                      boundIdxConst(0)},
      ArrayRef<Value>{boundIdxConst(first_dst_tile_sublane_offset),
                      boundIdxConst(target_shape[1])});
  return select_32bit(sublanes_mask, right_partial_vreg, left_partial_vreg);
}

// Retiles across vregs to match the destination layout when the sublane tiling
// dimension is reduced.
//
// Arguments:
//   value_shape: The shape of the value which needs to be retiled in vregs.
//   src: The source layout.
//   src_vreg_array: An array of vregs storing source tiles (with implicit
//                   shape).
//   dst_layout: The destination layout, with reduced sublane dimension, based
//   on
//     which the retiling will be performed.
//   hw_generation: The generation of a target hardware.
//
// Returns:
//   A new array of vregs that store tiles based on the destination layout.
xla::Array<Value> retileToReducedSublanes(
    OpBuilder &builder, const ArrayRef<int64_t> value_shape,
    const VectorLayout &src_layout, const xla::Array<Value> &src_vreg_array,
    const VectorLayout &dst_layout, const std::array<int64_t, 2> target_shape) {
  const int64_t dst_tiling_sublane = dst_layout.tiling()[0];
  CHECK_LT(0, dst_tiling_sublane);
  CHECK_LT(dst_tiling_sublane, src_layout.tiling()[0]);
  CHECK(llvm::isPowerOf2_64(dst_tiling_sublane));

  xla::Array<Value> dst_vreg_array(
      dst_layout.tileArrayImplicitShape(value_shape, target_shape));

  // We need to rotate each src tile in each src vreg once so that they can
  // be merged to form new vregs. If a src vreg contains more than one src tile,
  // it will be rotated once per src tile. Consider (8,512) tensor stored with
  // layout (8,128) in a vreg array of shape (1, 4). Each src vreg
  // contains one src tile in this case. Given, the destination layout is
  // (2,128), each src tile is divided into 4 destination tiles as shown below:
  //
  //   src_vreg_0_0:     src_vreg_0_1:    src_vreg_0_2:   src_vreg_0_3:
  // dst_tile_0_0_0    dst_tile_0_0_1   dst_tile_0_0_2  dst_tile_0_0_3
  // dst_tile_1_0_0    dst_tile_1_0_1   dst_tile_1_0_2  dst_tile_1_0_3
  // dst_tile_2_0_0    dst_tile_2_0_1   dst_tile_2_0_2  dst_tile_2_0_3
  // dst_tile_3_0_0    dst_tile_3_0_1   dst_tile_3_0_2  dst_tile_3_0_3
  //
  // In this example, each src tile in the src vreg is rotated by
  // col *  sublanes_per_tile to produce the following rotated src vregs:
  //
  // rot_src_vreg_0_0: rot_src_vreg_0_1: rot_src_vreg_0_2: rot_src_vreg_0_3:
  //     dst_tile_0_0_0    dst_tile_3_0_1    dst_tile_2_0_2    dst_tile_1_0_3
  //     dst_tile_1_0_0    dst_tile_0_0_1    dst_tile_3_0_2    dst_tile_2_0_3
  //     dst_tile_2_0_0    dst_tile_1_0_1    dst_tile_0_0_2    dst_tile_3_0_3
  //     dst_tile_3_0_0    dst_tile_2_0_1    dst_tile_1_0_2    dst_tile_0_0_3

  // If there were 2 src tiles in the src vreg, we would have rotated each src
  // vreg twice, producing 2 rotated src vreg per src vreg. The rotation amount
  // is calculated from the src and the dest tiling.

  const int64_t src_tiles_per_vreg = src_layout.tilesPerVreg(target_shape);
  const int64_t dst_tiles_per_vreg = dst_layout.tilesPerVreg(target_shape);
  const int64_t src_sublanes_per_tile =
      src_layout.sublanesPerTile(target_shape);
  const int64_t dst_sublanes_per_tile =
      dst_layout.sublanesPerTile(target_shape);
  // Each vreg may store more than one src tile. We may have to rotate a vreg,
  // once for every src tile in the vreg.
  SmallVector<int64_t> rotated_src_vreg_array_shape(
      toArrayRef(src_vreg_array.dimensions()));
  rotated_src_vreg_array_shape.back() *= src_tiles_per_vreg;
  xla::Array<Value> rotated_src_vreg_array(rotated_src_vreg_array_shape);

  rotated_src_vreg_array.Each([&](const absl::Span<const int64_t> rotated_idx,
                                  Value *const rotated_src_vreg) {
    const int64_t idx = rotated_idx.back();
    const int64_t tile_idx = idx % dst_tiles_per_vreg;
    const int64_t dst_sublane = tile_idx * dst_sublanes_per_tile;
    auto [src_col, src_tile_offset] = std::div(idx, src_tiles_per_vreg);
    SmallVector<int64_t> src_vreg_idx(toArrayRef(rotated_idx));
    src_vreg_idx.back() = src_col;
    Value src_vreg = src_vreg_array(src_vreg_idx);
    const int64_t src_sublane = src_tile_offset * src_sublanes_per_tile;
    int64_t rotate_amt = dst_sublane - src_sublane;
    if (rotate_amt == 0) {
      *rotated_src_vreg = src_vreg;
      return;
    }
    if (rotate_amt < 0) {
      rotate_amt += target_shape[0];
    }
    *rotated_src_vreg = builder.create<tpu::RotateOp>(
        src_vreg.getLoc(), src_vreg, rotate_amt,
        /*dimension=*/0, /*stride=*/nullptr, /*stride_dimension=*/nullptr);
  });
  // Assemble output vregs using tiles from rotated vregs using select.
  // Given, above example, destination vregs are then assembled as follows:
  //  dst_vreg_0_0:
  // dst_tile_0_0_0
  // dst_tile_0_0_1
  // dst_tile_0_0_2
  // dst_tile_0_0_3

  //  dst_vreg_1_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_1_0_3
  // dst_tile_1_0_0
  // dst_tile_1_0_1
  // dst_tile_1_0_2

  //  dst_vreg_2_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_2_0_2
  // dst_tile_2_0_3
  // dst_tile_2_0_0
  // dst_tile_2_0_1

  //  dst_vreg_3_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_3_0_1
  // dst_tile_3_0_2
  // dst_tile_3_0_3
  // dst_tile_3_0_0

  // Each destination vreg is assembled from destination tiles in multiple
  // rotated src vregs. In the above example, if we wanted each destination tile
  // to be in correct sublane offset in a rotated vreg, say rot_src_vreg_0_1,
  // before assembling the destination tiles, we would have had to rotate
  // src_vreg_0_1 four times, creating 4 rotated vregs (instead of 1) for each
  // src vreg. In the above example, we instead rotated a src vreg src_vreg_0_1
  // only once to obtain rot_src_vreg_0_1 where the dst_tile_0_0_1 is in correct
  // final sublane offset, i.e. 2. But notice the sublane offset of
  // dst_tile_1_0_1 in the same rotated vreg. Its correct final destination
  // sublane offset is 2, but in rot_src_vreg_0_1, its offset is 4. Its sublane
  // offset is off by 2. We need to correct these sublane offsets in the final
  // assembled dst vregs. A single rotation of each assembled dst vreg is needed
  // to correct such sublane offsets. This strategy reduces the number of
  // sublane rotations required. See comments below.
  const int64_t tile_sublane_change_factor =
      src_layout.tiling()[0] / dst_layout.tiling()[0];

  dst_vreg_array.Each([&](absl::Span<const int64_t> idx,
                          Value *const dst_vreg) {
    const int64_t row = *(idx.end() - 2);
    const int64_t col = *(idx.end() - 1);
    auto [rotated_vreg_row, first_dst_tile_offset] =
        std::div(row, tile_sublane_change_factor);
    const int64_t first_dst_tile_sublane_offset =
        first_dst_tile_offset * dst_sublanes_per_tile;
    const int64_t src_vreg_array_col_start = col * dst_tiles_per_vreg;
    const int64_t src_vreg_array_col_end =
        std::min((col + 1) * dst_tiles_per_vreg,
                 rotated_src_vreg_array.dimensions().back()) -
        1;

    // TODO(tlongeri): Find a better way to slice that doesn't involve so
    // copying so many index vectors and hopefully is more concise. Probably
    // by expanding xla::Array (maybe could just expose calculate_index?).
    SmallVector<int64_t> rotated_row_starts(toArrayRef(idx));
    *(rotated_row_starts.end() - 2) = rotated_vreg_row;
    *(rotated_row_starts.end() - 1) = 0;
    SmallVector<int64_t> rotated_row_ends(idx.size());
    for (size_t i = 0; i + 1 < rotated_row_ends.size(); ++i) {
      rotated_row_ends[i] = rotated_row_starts[i] + 1;
    }
    *(rotated_row_ends.end() - 1) = rotated_src_vreg_array.dimensions().back();
    const xla::Array<Value> rotated_row_slice =
        rotated_src_vreg_array.Slice(rotated_row_starts, rotated_row_ends);
    const Value dst_tile = selectTilesFromRotatedRowVregs(
        builder, /*rotated_row_vregs=*/
        ArrayRef(rotated_row_slice.begin(), rotated_row_slice.end()),
        src_vreg_array_col_start, src_vreg_array_col_end,
        first_dst_tile_sublane_offset, dst_layout, target_shape);
    if (first_dst_tile_sublane_offset == 0) {
      // No need to rotate. First dst tile is already at offset 0, which means
      // rest of the dst tiles are also at correct sublane offset.
      *dst_vreg = dst_tile;
    } else {
      // Fix the destination tile sublane offset by rotating assembled dest vreg
      // once (See comments above). The dst vregs are fixed as follows:
      // No rotation needed.
      // dst_tile_0_0_0
      // dst_tile_0_0_1
      // dst_tile_0_0_2
      // dst_tile_0_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=1):
      // dst_tile_1_0_0
      // dst_tile_1_0_1
      // dst_tile_1_0_2
      // dst_tile_1_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=2):
      // dst_tile_2_0_0
      // dst_tile_2_0_1
      // dst_tile_2_0_2
      // dst_tile_2_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=3):
      // dst_tile_3_0_0
      // dst_tile_3_0_1
      // dst_tile_3_0_2
      // dst_tile_3_0_3
      *dst_vreg = builder.create<tpu::RotateOp>(
          dst_tile.getLoc(), dst_tile,
          target_shape[0] - first_dst_tile_sublane_offset, /*dimension=*/0,
          /*stride=*/nullptr, /*stride_dimension=*/nullptr);
    }
  });
  return dst_vreg_array;
}


// Copy one sublane from a vreg to another vreg.
//
// Arguments:
//  src_vreg: The source vreg to copy a sublane from.
//  src_sl_idx: The sublane index in src_vreg to copy from.
//  dst_vreg: The base vreg to copy the sublane into. May be null.
//  dst_sl_idx: The sublane index in the result.
//
// Returns:
//  A new dst_vreg with the copied sublane.
Value copy_one_sublane(OpBuilder &builder, Value src_vreg, int src_sl_idx,
                       Value dst_vreg, int dst_sl_idx,
                       const std::array<int64_t, 2> target_shape) {
  src_vreg = builder.create<tpu::RotateOp>(
      src_vreg.getLoc(), src_vreg,
      /*amount=*/(dst_sl_idx - src_sl_idx + target_shape[0]) % target_shape[0],
      /*dimension=*/0, /*stride=*/nullptr, /*stride_dimension=*/nullptr);
  if (dst_vreg) {
    auto boundIdxConst =
        std::bind(IdxConst, std::placeholders::_1, builder, src_vreg.getLoc());
    const int bitwidth =
        cast<VectorType>(src_vreg.getType()).getElementTypeBitWidth();
    CHECK_EQ(bitwidth,
             cast<VectorType>(dst_vreg.getType()).getElementTypeBitWidth());
    const VectorType vmask_ty =
        getNativeVregOrVmaskType(builder.getI1Type(), bitwidth, target_shape);
    auto sublanes_mask = builder.create<tpu::CreateMaskOp>(
        src_vreg.getLoc(), vmask_ty,
        ValueRange{boundIdxConst(dst_sl_idx), boundIdxConst(0)},
        ValueRange{boundIdxConst(dst_sl_idx + 1),
                   boundIdxConst(target_shape[1])});
    src_vreg = builder.create<arith::SelectOp>(src_vreg.getLoc(), sublanes_mask,
                                               src_vreg, dst_vreg);
  }
  return src_vreg;
}

// This function is based on tpu_rotate_rule. It applies a shift of amount to
// a given dim. A major difference is that it "overflows", i.e. if the shift
// amount is such that it pushes us into a new vreg, we create a new vreg and
// fill it in with the remaining rows.
//
// The shift is the difference between layout_in and layout_out, on the
// given dim.
FailureOr<xla::Array<Value>> tpu_rotate_with_overflow(
    OpBuilder &builder, const std::array<int64_t, 2> target_shape,
    const Location loc, const VectorType vty, xla::Array<Value> in_tiles,
    int64_t dim, const VectorLayout &layout_in,
    const LayoutOffsets offsets_out) {
  if (!layout_in.hasNativeTiling(target_shape)) {
    return emitError(loc, "Not implemented: non-native tiling for layout");
  }
  if (layout_in.bitwidth() != 32) {
    return emitError(loc,
                     "Not implemented: multi-row shift with "
                     "bitwidth != 32");
  }
  // TODO(apaszke,mvoz): Just use offsets_out instead of this.
  VectorLayout layout_out(layout_in.bitwidth(), offsets_out, layout_in.tiling(),
                          layout_in.implicit_dim());

  int64_t tiling_dim = dim - (in_tiles.num_dimensions() - 2);
  if (tiling_dim != 0) {
    return emitError(loc,
                     "Rotate with overflow untested for "
                     "dim != 0");
  }
  auto amount =
      *layout_out.offsets()[tiling_dim] - *layout_in.offsets()[tiling_dim];

  SmallVector<int64_t> dst_tiles_shape =
      layout_out.tileArrayImplicitShape(vty.getShape(), target_shape);

  const VectorType res_vreg_ty =
      getNativeVregType(vty.getElementType(), target_shape);

  xla::Array<Value> out_tiles(dst_tiles_shape);

  // We update the result vregs in the following way:
  //  - If the offset is positive, write the first tile as is, if the offset
  //    is negative, blend it with the next tile.
  //  - Blend the rest of the tiles with the prior (positive offset) or next
  //    (negative offset) tile.
  //  - (In positive cases, we can get an extra vreg (overflow)) we write the
  //    remaining tiles.
  //    This only happens if the original input vreg size is smaller than the
  //    result vreg size (an offset) can "push" us into a new vreg.
  //
  //  Ex: (30, 128), starting offset 0, shift by 6, native tiling (8, 128)
  //  The input is (4, 1), where the first 3 vregs are full (0-24)
  //  and the last vreg is filled in rows 0-6. When we offset it by 6, we
  //  need a 4th vreg, as now vreg 0 is filled in 6-8 (2 total), vreg 1, 2, 3
  //  are filled in fully (8-16, 16-24, 24-32) (2 + 24 total), and vreg 4 is
  //  filled in 0-4. (2 + 24 + 4 = 30).

  // Negative offset amount means we:
  //
  //  Ex 1: (30, 128), input offset 6, shift by -2, native tiling (8, 128)
  //  (The result of the last example, for simplicity). In this case, we have
  //  (5, 1) vregs as decribed above. Because the shift does not cause us to
  //  shift back from the 5th vreg, we still need it. In such a case, the result
  //  vreg is still (5, 1).
  //
  //  - Write the first vreg as is.
  //  - The next vregs are blended with the prior one (except the last),
  //    where we blend by the shift amount. Ex: Vreg 1 goes from 6-8 to 4-8,
  //    pulling 2 rows from the next vreg.
  //  - The last tile is masked to only write the remaining rows.
  //    Ex: Vreg 4 goes from 0-4 to 0-2.
  //
  //  Ex 2: (30, 128), starting offset 6, shift by -6, native tiling (8, 128)
  //  In this case, we have (5, 1) vregs as described above. Because the shift
  //  causes us to shift back from the 5th vreg, we don't need it anymore.
  //  In such a case, the result vreg is (4, 1).
  //
  //  - All vregs are blended with the next one (except the last),
  //    where we blend by the shift amount. Ex: Vreg 1 goes from 6-8 to 0-8,
  //    pulling 6 rows from the next vreg.
  //  - The last tile is discarded - it was fully subsumed by the prior blends.
  //
  //  Ex 3: (30, 128), starting offset 0, shift by -6, native tiling (8, 128)
  //  In this case, we have (4, 1) vregs as described above.
  //  In such a case, the result vreg is (4, 1), where the first vreg is filled
  //  in rows 2-8 (6), and vregs 1 and 2 are filled in fully (8-16, 16-24), and
  //  vreg 3 is filled in rows 0-6.
  //
  //  NOTE - in such cases, where the abs(shift) in a negative shift > starting
  //  offset, we can actually implement this as a positive shift of the delta
  //  from the native tile size.
  //  in the example above, the delta is 8 - 6 + 0 = 2. The resulting vregs are
  //  the same as if we had shifted by 2, starting at offset 0.
  //
  //  Another example to demonstrate the point:
  //  Ex 4: (30, 128), starting offset 2, shift by -4, native tiling (8, 128)
  //  In this case, we start with (4, 1) vregs as described above.
  //  (2-8)(8-16)(16-24)(0-4). Shifting by -4 is the same as 8 - 4 + 2 = 6.
  //  So we can just shift by 6, starting at offset 0.
  //  Vreg 0 is filled in 6-8 (2 total), vreg 1, 2 and 3 are filled in fully
  //  (8-16, 16-24, 24-32) (2 + 24 total = 26) vreg 4 is filled with the
  //  remainder, 0-4 (30 total).
  //
  //  This means that no matter what the shift is, we should always
  //  rotate and compute the shift amount in such a way that the first input
  //  vreg is the first output vreg.

  // Compute the mask for the blend.
  // Positive blends blend "forward" and negative blends blend "backward".
  auto mask_val = amount;
  auto vreg_rot_amount = amount;
  if (amount < 0) {
    mask_val = layout_in.tiling()[tiling_dim] - std::abs(amount);
    vreg_rot_amount += target_shape[tiling_dim];
  }
  auto boundIdxConst = std::bind(IdxConst, std::placeholders::_1, builder, loc);
  auto mask = builder.create<tpu::CreateMaskOp>(
      loc, VectorType::get(target_shape, builder.getI1Type()),
      ValueRange{boundIdxConst(0), boundIdxConst(0)},
      ValueRange{boundIdxConst(mask_val), boundIdxConst(target_shape[1])});

  // Actually do the rotation.
  in_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
    if (dim >= in_tiles.num_dimensions() - 2) {
      *v = builder.create<tpu::RotateOp>(loc, res_vreg_ty, in_tiles(idxs),
                                         vreg_rot_amount, tiling_dim, nullptr,
                                         nullptr);
    }
  });

  // Walk the result tiles.
  // TODO(mvoz): There is a micro-optimization here where we can avoid
  // allocating blend indices per vreg.
  out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
    if (idxs[dim] == 0) {
      // A negative shift amount means we need to blend the first tile with the
      // next one, but only if we're not at the end of the input.
      if (amount < 0 && (idxs[dim] + 1 < in_tiles.dim(dim))) {
        SmallVector<int64_t> next_idx = {idxs.begin(), idxs.end()};
        next_idx[dim] = idxs[dim] + 1;
        *v = builder.create<arith::SelectOp>(loc, mask, in_tiles(idxs),
                                             in_tiles(next_idx));
      } else {
        // Positive shift, or negative shift at the end of the input.
        *v = in_tiles(idxs);
      }
    } else if (idxs[dim] < in_tiles.dim(dim)) {
      // write the rest as blended up to the end of the input
      if (amount < 0) {
        if (idxs[dim] + 1 < in_tiles.dim(dim)) {
          SmallVector<int64_t> next_idx = {idxs.begin(), idxs.end()};
          next_idx[dim] = idxs[dim] + 1;
          *v = builder.create<arith::SelectOp>(loc, mask, in_tiles(idxs),
                                               in_tiles(next_idx));
        } else {
          // Nothing to blend with, just write the last tile.
          *v = in_tiles(idxs);
        }
      } else {
        SmallVector<int64_t> prior_idx = {idxs.begin(), idxs.end()};
        prior_idx[dim] = idxs[dim] - 1;
        *v = builder.create<arith::SelectOp>(loc, mask, in_tiles(prior_idx),
                                             in_tiles(idxs));
      }
    } else {
      // write trailing if it's there (positive shift, increasing vreg count)
      // Use the last prior
      SmallVector<int64_t> prior_idx = {idxs.begin(), idxs.end()};
      prior_idx[dim] = idxs[dim] - 1;
      *v = in_tiles(prior_idx);
    }
  });

  return out_tiles;
}

void rotateVregs(OpBuilder &builder, xla::Array<Value> &vregs,
                 const int64_t amount, const int dimension) {
  if (amount != 0) {
    vregs.Each([&](absl::Span<const int64_t> idx, Value *vreg) {
      CHECK(vreg);
      *vreg = builder
                  .create<tpu::RotateOp>(vreg->getLoc(), *vreg,
                                         /*amount=*/amount,
                                         /*dimension=*/dimension,
                                         /*stride=*/nullptr,
                                         /*stride_dimension=*/nullptr)
                  .getResult();
    });
  }
};

void rotateSublanes(OpBuilder &builder, xla::Array<Value> &vregs,
                    const int64_t amount) {
  rotateVregs(builder, vregs, amount, 0);
}

void rotateLanes(OpBuilder &builder, xla::Array<Value> &vregs,
                 const int64_t amount) {
  rotateVregs(builder, vregs, amount, 1);
}

// Relayout src_vregs from layout src to layout dst, where dst is the same as
// src except that the column offset is dst_col_offset.
FailureOr<xla::Array<Value>> doColumnShiftRelayout(
    OpBuilder &builder, const ArrayRef<int64_t> shape,
    xla::Array<Value> src_vregs, const VectorLayout &src,
    const int64_t dst_col_offset, const std::array<int64_t, 2> target_shape) {
  CHECK(src.offsets()[1]);
  const std::array<int64_t, 2> tiled_ishape =
      src.getImplicitTiledDims(shape, 1);
  const Location loc = src_vregs.begin()->getLoc();
  const std::array<int64_t, 2> tiling = src.tiling();
  const std::array<int64_t, 2> vreg_slice = src.vregSlice(target_shape);
  const int bitwidth = src.bitwidth();
  const int packing = src.packing();
  const int64_t col_diff = dst_col_offset - *src.offsets()[1];
  if (tiling[0] % packing != 0 || tiling[1] != target_shape[1]) {
    return emitError(loc,
                     "Not implemented: Unsupported tiling for column shift");
  }
  // When shifting columns with multiple tiles per vreg, the overflowing
  // columns of a tile move to the next tile, and they have to be shifted
  // down. For example, for a 32-bit layout with (2, 128 tiling), when shifting
  // a vreg right by 138 (128 + 10):
  //
  //  +---------------+---------+    +---------+---------------+
  //  |   0:118       | 118:128 |    |-138:-128|      -128:-10 |
  //  +---------------+---------+    +---------+---------------+
  //  | 128:246       | 246:256 |    | -10:0   |         0:118 |
  //  +---------------+---------+ -> +---------+---------------+
  //  | 256:382       | 382:392 |    | 118:128 |       128:246 |
  //  +---------------+---------+    +---------+---------------+
  //  | 392:502       | 502:512 |    | 246:256 |       256:382 |
  //  +---------------+---------+    +---------+---------------+
  //
  // The negative numbers above are used for column intervals coming from the
  // previous vreg (if there is one).
  //
  // We can break the result vreg down into four parts:
  //
  //  +---------+---------------+
  //  | UL      | UR            |
  //  +         +---------------+
  //  |         | LR            |
  //  +---------+               +
  //  | LL      |               |
  //  +         +               +
  //  |         |               |
  //  +---------+---------------+
  //
  // Our example shifts right, which causes the upper parts to come from the
  // previous (along the minor dim) vreg of the array (if it exists) and the
  // lower parts to come from the original "current" vreg.
  //
  // - LR (Lower Right) comes from the current vreg lane-rotated by 10, and
  //   sublane-rotated down by 2 (1 tile).
  // - LL (Lower Left) comes from the current vreg lane-rotated by 10, and
  //   sublane-rotated down by 4 (2 tiles).
  // - UR (Upper Right) comes from the previous vreg lane-shifted by 10, and
  //   sublane-rotated down by 2 (1 tile).
  // - UL (Upper Left) comes from the previous vreg lane-shifted by 10, and
  //   sublane-rotated down by 4 (2 tiles).
  //
  // This partitioning also works similarly for left shifts, except that the
  // upper parts come from the current vreg, and the lower parts come from the
  // next vreg.
  //
  // In general, for any tiling and shift amount, we will partition the result
  // vreg into four like we did here. However, for some tilings and shift
  // amounts, some of the partitions may be empty. There are some notable cases:
  //
  // - Tile-aligned shifts result in empty left parts.
  // - Native tiling (a single tile per vreg) results in empty upper right and
  //   lower left parts.
  // - Shifts right by less than 1 tile result in empty upper right parts, and
  //   shifts left by less than 1 tile result in empty lower left parts.

  const int64_t sublanes_per_tile = src.sublanesPerTile(target_shape);
  const int64_t tiles_per_vreg = src.tilesPerVreg(target_shape);

  int64_t split_offset = col_diff;
  int64_t upper_idx_delta = -1;
  int64_t lower_idx_delta = 0;
  if (col_diff < 0) {
    split_offset += vreg_slice[1];
    ++upper_idx_delta;
    ++lower_idx_delta;
  }
  const int64_t left_tile_split = llvm::divideCeil(split_offset, tiling[1]);
  const int64_t right_tile_split = split_offset / tiling[1];
  const int64_t left_right_split = split_offset % tiling[1];

  rotateLanes(builder, src_vregs, left_right_split);
  // TODO(tlongeri): Clean up. Some of these rotations may end up unused:
  // - The left part of the first vreg and the right part of the last vreg
  //   may be entirely padding.
  // - The entire left part may be unused if the shift is tile-aligned.
  // They will be removed as dead code anyway, but it would be nicer to not
  // generate them in the first place.
  // Also, sometimes the rotation amount is 0, so we don't need to allocate
  // another array (and we should steal the allocation for src_tiles, too).
  xla::Array<Value> left_part = src_vregs;
  xla::Array<Value> right_part = src_vregs;
  rotateSublanes(builder, left_part,
                 left_tile_split * sublanes_per_tile % target_shape[0]);
  rotateSublanes(builder, right_part,
                 right_tile_split * sublanes_per_tile % target_shape[0]);
  // We assemble left and right, and then put them together.
  // TODO(tlongeri): Lower and upper first is probably better, it can be
  // reused for consecutive vregs. We can assemble lower_left+lower_right
  // for one vreg and upper_left+upper_right for the next one in the same
  // vselect. But the mask for assembling upper+lower is not as simple, so
  // it might be a bit more expensive to generate. Worth it for large vreg
  // arrays, I'm not sure about small ones (especially in older TPU gens).
  const auto mask_vreg_ty = VectorType::get(
      packing == 1
          ? target_shape
          : ArrayRef<int64_t>{target_shape[0], target_shape[1], packing},
      builder.getI1Type());
  Value left_mask = nullptr;
  Value right_mask = nullptr;
  Value left_right_mask = nullptr;
  auto get_left_mask = [&]() {
    if (left_mask == nullptr) {
      left_mask = builder.create<tpu::CreateMaskOp>(
          loc, mask_vreg_ty,
          ArrayRef<Value>{IdxConst(0, builder, loc), IdxConst(0, builder, loc)},
          ArrayRef<Value>{
              IdxConst(left_tile_split * sublanes_per_tile, builder, loc),
              IdxConst(target_shape[1], builder, loc)});
    }
    return left_mask;
  };
  auto get_right_mask = [&]() {
    if (right_mask == nullptr) {
      right_mask = builder.create<tpu::CreateMaskOp>(
          loc, mask_vreg_ty,
          ArrayRef<Value>{IdxConst(0, builder, loc), IdxConst(0, builder, loc)},
          ArrayRef<Value>{
              IdxConst(right_tile_split * sublanes_per_tile, builder, loc),
              IdxConst(target_shape[1], builder, loc)});
    }
    return right_mask;
  };
  auto get_left_right_mask = [&]() {
    if (left_right_mask == nullptr) {
      left_right_mask = builder.create<tpu::CreateMaskOp>(
          loc, mask_vreg_ty,
          ArrayRef<Value>{IdxConst(0, builder, loc), IdxConst(0, builder, loc)},
          ArrayRef<Value>{IdxConst(target_shape[0], builder, loc),
                          IdxConst(left_right_split, builder, loc)});
    }
    return left_right_mask;
  };
  xla::Array<Value> dst_vregs(VectorLayout(bitwidth,
                                           {src.offsets()[0], dst_col_offset},
                                           tiling, src.implicit_dim())
                                  .tileArrayImplicitShape(shape, target_shape));
  dst_vregs.Each([&](absl::Span<const int64_t> dst_idx, Value *dst_vreg) {
    SmallVector<int64_t> dst_idx_local(toArrayRef(dst_idx));
    Value lower_left = nullptr;
    Value lower_right = nullptr;
    Value upper_left = nullptr;
    Value upper_right = nullptr;
    // Set parts if their size is non-empty and the source vreg exists.
    *(dst_idx_local.end() - 1) += lower_idx_delta;
    if (*(dst_idx_local.end() - 1) < *(src_vregs.dimensions().end() - 1)) {
      if (left_tile_split < tiles_per_vreg && 0 < left_right_split) {
        lower_left = left_part(dst_idx_local);
      }
      if (right_tile_split < tiles_per_vreg) {
        lower_right = right_part(dst_idx_local);
      }
    }
    *(dst_idx_local.end() - 1) -= lower_idx_delta;
    *(dst_idx_local.end() - 1) += upper_idx_delta;
    if (*(dst_idx_local.end() - 1) >= 0) {
      if (0 < left_tile_split && 0 < left_right_split) {
        upper_left = left_part(dst_idx_local);
      }
      if (0 < right_tile_split) {
        upper_right = right_part(dst_idx_local);
      }
    }
    *(dst_idx_local.end() - 1) -= upper_idx_delta;

    // For the first and last vregs, some parts may be all padding, so
    // unset them if this is the case. Note that the first and last vreg
    // are the same when there is only one.
    if (*(dst_idx_local.end() - 1) == 0) {
      // We check the final offset (note that this is different from the rotate
      // amount) against the thresholds of the last columns of vreg parts.
      if (right_tile_split * tiling[1] <= dst_col_offset) {
        // Note: When shifting right, UR is always all-padding.
        upper_right = nullptr;
      }
      if (split_offset <= dst_col_offset) {
        // Note: When shifting right, UL is always all-padding. When shifting
        // left, UL is never all-padding (unless this is also the last vreg,
        // possibly).
        upper_left = nullptr;
      }
      if (vreg_slice[1] - tiling[1] + left_right_split <= dst_col_offset) {
        // Note: When shifting right, LL is only all-padding if the source
        // offset is in the last tile. When shifting left, LL is never
        // all-padding (unless this is also the last vreg, possibly).
        lower_left = nullptr;
      }
    }
    if (*(dst_idx_local.end() - 1) == *(dst_vregs.dimensions().end() - 1) - 1) {
      // We check the final end offset against the thresholds of the first
      // columns of vreg parts.
      const uint64_t end_offset =
          (dst_col_offset + tiled_ishape[1] - 1) % vreg_slice[1] + 1;
      if (end_offset <= left_tile_split * tiling[1]) {
        // Note: When shifting left, LL is always all-padding.
        lower_left = nullptr;
      }
      if (end_offset <= split_offset) {
        // Note: When shifting left, LR is always all-padding. When shifting
        // right, LR is never all-padding (unless this is also the first vreg,
        // possibly).
        lower_right = nullptr;
      }
      if (end_offset <= left_right_split) {
        // Note: When shifting left, UR is only all-padding if the original
        // end offset is in the first tile. When shifting right, UR is never
        // all-padding (unless this is also the last vreg, possibly).
        upper_right = nullptr;
      }
    }
    // Combine parts into the final vreg (see comment in mask definitions).
    auto combine_parts = [&builder](Value part1, Value part2,
                                    auto get_mask_fn) -> Value {
      if (part1 && part2) {
        return builder.create<arith::SelectOp>(part1.getLoc(), get_mask_fn(),
                                               part1, part2);
      } else if (part1) {
        return part1;
      } else {
        return part2;
      }
    };
    Value left = combine_parts(upper_left, lower_left, get_left_mask);
    Value right = combine_parts(upper_right, lower_right, get_right_mask);
    *dst_vreg = combine_parts(left, right, get_left_right_mask);
    CHECK(*dst_vreg);
  });
  return dst_vregs;
}

FailureOr<std::pair<VectorLayout, xla::Array<Value>>> changeOffsets(
    RewriteContext &ctx, OpBuilder &builder, const Location loc,
    const VectorType vty, const VectorLayout src, xla::Array<Value> vregs,
    const LayoutOffsets dst_offsets) {
  const auto &target_shape = ctx.target_shape;
  const VectorLayout dst(src.bitwidth(), dst_offsets, src.tiling(),
                         src.implicit_dim());
  const int packing = src.packing();
  const int8_t bitwidth = src.bitwidth();

  int row_diff;
  if (!src.offsets()[0].has_value()) {
    row_diff = 0;
  } else if (!dst_offsets[0].has_value()) {
    return emitError(loc, "Not implemented: Sublane broadcast");
  } else {
    row_diff = *dst_offsets[0] - *src.offsets()[0];
  }

  int64_t col_diff;
  if (!src.offsets()[1].has_value()) {
    col_diff = 0;
  } else if (!dst_offsets[1].has_value()) {
    return emitError(loc, "Not implemented: Lane broadcast");
  } else {
    col_diff = *dst_offsets[1] - *src.offsets()[1];
  }

  if (row_diff != 0) {
    if (col_diff != 0) {
      return emitError(loc, "Not implemented: Row and column offset changes");
    }
    const SmallVector<int64_t> implicit_shape =
        src.implicitShape(vty.getShape());
    if (implicit_shape[implicit_shape.size() - 2] != 1) {
      // Multi row shift
      // TODO(mvoz): This should take the vregs array, not the value.
      FAILUREOR_ASSIGN_OR_RETURN(
          vregs, tpu_rotate_with_overflow(
                     builder, target_shape, loc, vty, std::move(vregs),
                     /*dim*/ implicit_shape.size() - 2, src, dst_offsets));
    } else {
      // Single row case
      // TODO(mvoz): The single row case has a broader set of supported
      // operations: non-native tiling, packed types, implicit dim. We should
      // support these cases in tpu_rotate_with_overflow and remove this
      // branch.
      const int64_t src_sublane = *src.offsets()[0] / packing;
      const int64_t dst_sublane = *dst_offsets[0] / packing;
      if (int64_t sublane_diff = dst_sublane - src_sublane) {
        if (sublane_diff < 0) {
          sublane_diff += target_shape[0];
        }
        rotateSublanes(builder, vregs, sublane_diff);
      }
      const int src_subelem = *src.offsets()[0] % packing;
      const int dst_subelem = *dst.offsets()[0] % packing;
      if (src_subelem != dst_subelem) {
        const int subelem_diff = dst_subelem - src_subelem;
        const int shift_bits = bitwidth * std::abs(subelem_diff);
        VectorType bits_vreg_ty =
            VectorType::get(target_shape, builder.getI32Type());
        auto shift_vreg = builder.create<arith::ConstantOp>(
            loc, bits_vreg_ty,
            DenseElementsAttr::get(bits_vreg_ty, shift_bits));
        vregs.Each([&](absl::Span<const int64_t> /*idx*/, Value *tile) {
          auto bit_tile =
              builder.create<tpu::BitcastVregOp>(loc, bits_vreg_ty, *tile);
          Operation *shift_tile;
          if (subelem_diff > 0) {
            shift_tile =
                builder.create<arith::ShLIOp>(loc, bit_tile, shift_vreg);
          } else {  // subelem_diff < 0
            CHECK_LT(subelem_diff, 0);
            shift_tile =
                builder.create<arith::ShRUIOp>(loc, bit_tile, shift_vreg);
          }
          *tile = builder
                      .create<tpu::BitcastVregOp>(loc, tile->getType(),
                                                  shift_tile->getResult(0))
                      .getResult();
        });
      }
    }
  }

  // Rows are now correctly aligned. Time to offset columns.
  // TODO(apaszke, mvoz): Changing an offset might add or remove one vreg.
  // Note - this is handled for row shifts via tpu_rotate_with_overflow
  SmallVector<int64_t> dst_tiles_shape =
      dst.tileArrayImplicitShape(vty.getShape(), target_shape);
  CHECK_EQ(*(dst_tiles_shape.end() - 2), *(vregs.dimensions().end() - 2));

  // TODO(tlongeri): Clean up col_diff and pass the dst offset directly.
  if (col_diff != 0) {
    FAILUREOR_ASSIGN_OR_RETURN(
        vregs, doColumnShiftRelayout(builder, vty.getShape(), std::move(vregs),
                                     src, *dst.offsets()[1], target_shape));
  }
  return std::make_pair(dst, std::move(vregs));
}

LogicalResult retileToLargeTileWithScratch(
    RewriteContext &ctx, OpBuilder &builder, const Location loc,
    xla::Array<Value> &dst_tiles, const std::array<int64_t, 2> &dst_tile,
    const xla::Array<Value> &src_tiles, const std::array<int64_t, 2> &src_tile,
    TypedValue<MemRefType> scratch_ref, const int64_t store_vreg_delay,
    const int64_t load_vreg_skips) {
  if (dst_tile[0] % src_tile[0] != 0) {
    return failure();
  }
  // Number of src vregs needed to assemble one dst vreg.
  int vregs_per_group = dst_tile[0] / src_tile[0];
  // Number of sublanes needed per src vreg to assemble one dst vreg.
  int sl_per_vreg = ctx.target_shape[0] / vregs_per_group;
  int stride = vregs_per_group;

  xla::Array<int> sublane_offsets(
      {ctx.target_shape[0] / dst_tile[0], src_tile[0], vregs_per_group}, 0);
  absl::c_iota(sublane_offsets, 0);
  // The older hardware has limited support for shuffles so even if we have bank
  // conflicts, we just accept them and will have the lowering unroll the
  // loads/stores.
  int64_t num_offsets = sublane_offsets.num_elements();
  // The max sublane offset before handling bank conflicts is always
  // (num_offsets - 1). To avoid bank conflicts, we need to add one extra
  // sublane to stride so (num_offsets - 1) / stride is the extra offset needed
  // to pad sublanes.
  //
  // For example, if store stride = 4, sublane_count = 8, and
  // load offsets = [0, 1, 2, 3, 4, 5, 6, 7], then the sublane offsets after
  // handling bank conflicts will be [0, 1, 2, 3, 5, 6, 7, 8] and the max
  // sublane offset will be 7 + (8 - 1) / 4 = 8.
  //
  // Before
  //        <-------- sublanes --------->
  //        0    1  ...                 32
  // store: x---x---x---x---x---x---x---x
  // load:  xxxxxxxxx--------------------
  //
  // After
  //        <-------- sublanes --------->
  //        0    5  ...                        40
  // store: x----x----x----x----x----x----x----x
  // load:  xxxx-xxxx---------------------------
  //
  // where "x" indicates a sublane that needs to be accessed and "-"" indicates
  // a sublane that does not need to be accessed.
  int max_sublane_offset = (num_offsets - 1) + (num_offsets - 1) / stride;
  bool should_handle_bank_confict =
      shouldHandleBankConflict(ctx, stride, max_sublane_offset);
  if (should_handle_bank_confict) {
    handleBankConflict(stride, absl::MakeSpan(sublane_offsets.data(),
                                              sublane_offsets.num_elements()));
  }
  sublane_offsets.TransposeDimensions({0, 2, 1});

  auto mlirIndexConst = [&](int d) {
    return builder.create<arith::ConstantOp>(
        src_tiles.begin()->getLoc(),
        builder.getIntegerAttr(builder.getIndexType(), d));
  };
  auto cst_0 = mlirIndexConst(0);
  // Each group has exact number of src vregs needed to assemble one dst vreg.
  // We can not use circular buffer here because we need to have enough space to
  // strided load/store.
  int64_t sublanes_per_group = stride * sl_per_vreg * vregs_per_group;
  int64_t max_groups_in_scratch =
      ctx.max_sublanes_in_scratch / sublanes_per_group;
  if (max_groups_in_scratch < 1) {
    return emitError(loc,
                     "scratch space is not enough for retiling to large tile");
  }
  int64_t stored_group_cnt = 0;
  auto dst_vreg_ty = src_tiles.begin()->getType();
  // Create a new vreg type that can be stored in scratch memref.
  auto temp_vreg_ty =
      VectorType::get(ctx.target_shape, scratch_ref.getType().getElementType());
  SmallVector<bool, 8> sublane_mask(ctx.target_shape[0], true);
  // (dst_vreg, load_offset)
  std::vector<std::pair<Value *, Value>> delayed_loads;
  delayed_loads.reserve(max_groups_in_scratch * vregs_per_group);
  // We only emit the loads when we run out of scratch space or we are at the
  // last vreg of the batch to help bundle scheduling.
  auto emit_all_delayed_loads = [&]() {
    for (auto [dst_vreg, load_offset] : delayed_loads) {
      Value load_op = builder.create<tpu::ShuffledLoadOp>(
          loc, temp_vreg_ty, scratch_ref, ArrayRef<Value>({load_offset, cst_0}),
          ArrayRef<bool>(sublane_mask),
          ArrayRef<int32_t>(sublane_offsets.begin(), sublane_offsets.end()));
      *dst_vreg = builder.create<tpu::BitcastVregOp>(loc, dst_vreg_ty, load_op);
    }
    delayed_loads.clear();
  };

  int rank = src_tiles.dimensions().size();
  if (rank != dst_tiles.dimensions().size()) {
    return emitError(loc, "src and dst tiles have different ranks");
  }
  for (int i = 0; i < rank - 2; ++i) {
    if (src_tiles.dim(i) != dst_tiles.dim(i)) {
      return emitError(loc,
                       "Expected src and dst tiles have same dimension "
                       "sizes on dim")
             << i << ", but got " << src_tiles.dim(i) << " vs "
             << dst_tiles.dim(i);
    }
  }
  SmallVector<int64_t, 4> src_idx(rank);
  dst_tiles.Each([&](absl::Span<const int64_t> dst_idx, Value *dst_vreg) {
    int64_t dst_row_idx = *(dst_idx.end() - 2);
    int64_t dst_col_idx_with_skips = *(dst_idx.end() - 1) + load_vreg_skips;
    int64_t vreg_idx_in_group = dst_col_idx_with_skips % vregs_per_group;
    int64_t load_offset = sublanes_per_group * stored_group_cnt +
                          vreg_idx_in_group * sl_per_vreg * stride;
    delayed_loads.push_back(
        std::make_pair(dst_vreg, mlirIndexConst(load_offset)));
    // When dst vreg is at the last vreg of the group or the current dst
    // vregs' row, this indicates we have scheduled delayed loads for all
    // the vregs from current group and now we need to store corresponding
    // group of src vregs before actually emitting the loads.
    if (vreg_idx_in_group == vregs_per_group - 1 ||
        dst_idx.back() == dst_tiles.dimensions().back() - 1) {
      auto base_src_row_idx = dst_row_idx * vregs_per_group - store_vreg_delay;
      auto src_col_idx = dst_col_idx_with_skips / vregs_per_group;
      std::copy(dst_idx.begin(), dst_idx.end(), src_idx.begin());
      for (int vi = 0; vi < vregs_per_group; ++vi) {
        const int64_t src_row_idx = base_src_row_idx + vi;
        if (src_row_idx < 0) {
          continue;
        }
        if (src_row_idx >= src_tiles.dim(rank - 2) ||
            src_col_idx >= src_tiles.dim(rank - 1)) {
          break;
        }
        *(src_idx.end() - 2) = src_row_idx;
        *(src_idx.end() - 1) = src_col_idx;
        Value src_vreg = src_tiles(src_idx);
        src_vreg =
            builder.create<tpu::BitcastVregOp>(loc, temp_vreg_ty, src_vreg);
        Value store_offset =
            mlirIndexConst(sublanes_per_group * stored_group_cnt + vi);
        builder.create<tpu::StoreOp>(
            loc, src_vreg, scratch_ref, ArrayRef<Value>({store_offset, cst_0}),
            ArrayRef<bool>(sublane_mask),
            /*mask=*/nullptr, builder.getI32IntegerAttr(stride));
      }
      stored_group_cnt = (stored_group_cnt + 1) % max_groups_in_scratch;
      // We emit loads when we run out of scratch space or we are at the
      // last vreg of the batch.
      if (stored_group_cnt == 0 ||
          (*(dst_idx.end() - 2) == dst_tiles.dim(rank - 2) - 1 &&
           *(dst_idx.end() - 1) == dst_tiles.dim(rank - 1) - 1)) {
        emit_all_delayed_loads();
      }
    }
  });
  return success();
}

LogicalResult retileToSmallTileWithScratch(
    RewriteContext &ctx, OpBuilder &builder, const Location loc,
    xla::Array<Value> &dst_tiles, const std::array<int64_t, 2> &dst_tile,
    const xla::Array<Value> &src_tiles, const std::array<int64_t, 2> &src_tile,
    TypedValue<MemRefType> scratch_ref, const int64_t store_vreg_delay,
    const int64_t load_vreg_skips) {
  if (src_tile[0] % dst_tile[0] != 0) {
    return failure();
  }
  // Number of src vregs needed to assemble one dst vreg.
  int vregs_per_group = src_tile[0] / dst_tile[0];
  // Number of sublanes needed per src vreg to assemble one dst vreg.
  int sl_per_vreg = ctx.target_shape[0] / vregs_per_group;
  int stride = vregs_per_group;

  xla::Array<int> sublane_offsets(
      {ctx.target_shape[0] / src_tile[0], dst_tile[0], vregs_per_group}, 0);
  absl::c_iota(sublane_offsets, 0);
  // The older hardware has limited support for shuffles so even if we have
  // bank conflicts, we just accept them and will have the lowering unroll the
  // loads/stores.
  int64_t num_offsets = sublane_offsets.num_elements();
  // The max sublane offset before handling bank conflicts is always
  // (num_offsets - 1). To avoid bank conflicts, we need to add one extra
  // sublane to stride so (num_offsets - 1) / stride is the extra offset needed
  // to pad sublanes.
  //
  // For example, if store stride = 4, sublane_count = 8, and
  // load offsets = [0, 1, 2, 3, 4, 5, 6, 7], then the sublane offsets after
  // handling bank conflicts will be [0, 1, 2, 3, 5, 6, 7, 8] and the max
  // sublane offset will be 7 + (8 - 1) / 4 = 8.
  //
  // Before
  //        <-------- sublanes --------->
  //        0   4   ...
  // store: x---x---x---x---x---x---x---x
  // load:  xxxxxxxxx-------------------
  //
  // After
  //        <-------- sublanes --------->
  //        0    5  ...
  // store: x----x----x----x----x----x----x----x
  // load:  xxxx-xxxx---------------------------
  //
  // where "x" indicates a sublane that needs to be accessed and "-"" indicates
  // a sublane that does not need to be accessed.
  int max_sublane_offset = (num_offsets - 1) + (num_offsets - 1) / stride;
  bool should_handle_bank_confict =
      shouldHandleBankConflict(ctx, stride, max_sublane_offset);
  bool use_shuffled_load = false;
  if (ctx.hardware_generation <= 4) {
    if (src_tile[0] == 8) {
      // The older hardware does not support shuffled store. However, if the src
      // tile is (8, 128), we can convert (shuffled store + strided load) to
      // (strided store + shuffled load).
      use_shuffled_load = true;
    } else if (src_tile[0] == 4) {
      // In this case, the trick of replacing a shuffled store with a shuffled
      // load does not work. Handling bank conflicts will cause the sublane
      // offsets to increase which might make emulation harder, so we avoid
      // doing so.
      should_handle_bank_confict = false;
    }
  }

  // Add one extra sublane to stride to avoid bank conflict.
  if (should_handle_bank_confict) {
    handleBankConflict(stride, absl::MakeSpan(sublane_offsets.data(),
                                              sublane_offsets.num_elements()));
  }
  sublane_offsets.TransposeDimensions({0, 2, 1});
  auto mlirIndexConst = [&](int d) {
    return builder.create<arith::ConstantOp>(
        src_tiles.begin()->getLoc(),
        builder.getIntegerAttr(builder.getIndexType(), d));
  };
  auto cst_0 = mlirIndexConst(0);
  // Each group has exact number of src vregs needed to assemble one dst vreg.
  // We can not use circular buffer here because we need to have enough space
  // to strided load/store.
  int64_t sublanes_per_group = stride * sl_per_vreg * vregs_per_group;
  int64_t max_groups_in_scratch =
      ctx.max_sublanes_in_scratch / sublanes_per_group;
  if (max_groups_in_scratch < 1) {
    return emitError(loc,
                     "scratch space is not enough for retiling to small tile");
  }
  int64_t stored_group_cnt = 0;
  auto dst_vreg_ty = src_tiles.begin()->getType();
  // Create a new vreg type that can be stored in scratch memref.
  auto temp_vreg_ty =
      VectorType::get(ctx.target_shape, scratch_ref.getType().getElementType());
  SmallVector<bool, 8> sublane_mask(ctx.target_shape[0], true);
  // (dst_vreg, load_offset)
  std::vector<std::pair<Value *, Value>> delayed_loads;
  delayed_loads.reserve(max_groups_in_scratch * vregs_per_group);
  // We only emit the loads when we run out of scratch space or we are at the
  // last vreg of the batch to help bundle scheduling.
  auto emit_all_delayed_loads = [&]() {
    for (auto [dst_vreg, load_offset] : delayed_loads) {
      Value load_op;
      if (use_shuffled_load) {
        load_op = builder.create<tpu::ShuffledLoadOp>(
            loc, temp_vreg_ty, scratch_ref,
            ArrayRef<Value>({load_offset, cst_0}), ArrayRef<bool>(sublane_mask),
            ArrayRef<int32_t>(sublane_offsets.begin(), sublane_offsets.end()));
      } else {
        load_op = builder.create<tpu::LoadOp>(
            loc, temp_vreg_ty, scratch_ref,
            ArrayRef<Value>({load_offset, cst_0}), ArrayRef<bool>(sublane_mask),
            builder.getI32IntegerAttr(stride));
      }
      *dst_vreg = builder.create<tpu::BitcastVregOp>(loc, dst_vreg_ty, load_op);
    }
    delayed_loads.clear();
  };
  int rank = src_tiles.dimensions().size();
  if (rank != dst_tiles.dimensions().size()) {
    return emitError(loc, "src and dst tiles have different ranks");
  }
  for (int i = 0; i < rank - 2; ++i) {
    if (src_tiles.dim(i) != dst_tiles.dim(i)) {
      return emitError(loc,
                       "Expected src and dst tiles have same dimension "
                       "sizes on dim")
             << i << ", but got " << src_tiles.dim(i) << " vs "
             << dst_tiles.dim(i);
    }
  }
  SmallVector<int64_t, 4> dst_idx(rank);
  src_tiles.Each([&](absl::Span<const int64_t> src_idx, Value src_vreg) {
    int64_t src_row_idx = *(src_idx.end() - 2);
    int64_t src_col_idx_with_delays = *(src_idx.end() - 1) + store_vreg_delay;
    int64_t vreg_idx_in_group = src_col_idx_with_delays % vregs_per_group;
    src_vreg = builder.create<tpu::BitcastVregOp>(loc, temp_vreg_ty, src_vreg);
    if (use_shuffled_load) {
      Value store_offset = mlirIndexConst(
          sublanes_per_group * stored_group_cnt + vreg_idx_in_group);
      builder.create<tpu::StoreOp>(
          loc, src_vreg, scratch_ref, ArrayRef<Value>({store_offset, cst_0}),
          ArrayRef<bool>(sublane_mask),
          /*mask=*/nullptr, builder.getI32IntegerAttr(stride));
    } else {
      Value store_offset =
          mlirIndexConst(sublanes_per_group * stored_group_cnt +
                         vreg_idx_in_group * sl_per_vreg * stride);
      builder.create<tpu::ShuffledStoreOp>(
          loc, src_vreg, scratch_ref, ArrayRef<Value>({store_offset, cst_0}),
          ArrayRef<bool>(sublane_mask),
          ArrayRef<int32_t>(sublane_offsets.begin(), sublane_offsets.end()));
    }
    // When src vreg is at the last vreg of the group or the current src
    // vregs' row, this indicates we have stored all the vregs needed to
    // assemble a new group of dst vreg.
    if (vreg_idx_in_group == vregs_per_group - 1 ||
        src_idx.back() == src_tiles.dimensions().back() - 1) {
      auto base_dst_row_idx = src_row_idx * vregs_per_group - load_vreg_skips;
      auto dst_col_idx = src_col_idx_with_delays / vregs_per_group;
      std::copy(src_idx.begin(), src_idx.end(), dst_idx.begin());
      for (int vi = 0; vi < vregs_per_group; ++vi) {
        const int64_t dst_row_idx = base_dst_row_idx + vi;
        if (dst_row_idx < 0) {
          continue;
        }
        if (dst_row_idx >= dst_tiles.dim(rank - 2) ||
            dst_col_idx >= dst_tiles.dim(rank - 1)) {
          break;
        }
        *(dst_idx.end() - 2) = dst_row_idx;
        *(dst_idx.end() - 1) = dst_col_idx;
        Value *dst_vreg = &dst_tiles(dst_idx);
        int64_t load_offset =
            use_shuffled_load ? (sublanes_per_group * stored_group_cnt +
                                 vi * sl_per_vreg * stride)
                              : (sublanes_per_group * stored_group_cnt + vi);
        delayed_loads.push_back(
            std::make_pair(dst_vreg, mlirIndexConst(load_offset)));
      }
      stored_group_cnt = (stored_group_cnt + 1) % max_groups_in_scratch;
      // We emit loads when we run out of scratch space or we are at the
      // last vreg of the batch.
      if (stored_group_cnt == 0 ||
          (*(src_idx.end() - 2) == src_tiles.dim(rank - 2) - 1 &&
           *(src_idx.end() - 1) == src_tiles.dim(rank - 1) - 1)) {
        emit_all_delayed_loads();
      }
    }
  });
  return success();
}

// go/mosaic-retiling-in-scratch is the full internal documentation that
// includes more details about the TPU generations.
// Arguments:
// - shape:            The non-implicit shape of the operand
// - dst_tiling:       The desired result tiling
// - dst_offsets_hint: Hints for the result offsets. They may be used or
//                     ignored. See comments in the body of the function for
//                     more details.
// - src_vregs:        The source vregs to retile.
// - src:              The source layout
// Returns a pair holding the result layout (potentially using the hints) and
// the retiled vregs.
// TODO(tlongeri): Clean up the function parameters/signatures. We are passing
// in more information than strictly needed.
FailureOr<std::pair<VectorLayout, xla::Array<Value>>> retileWithScratch(
    RewriteContext &ctx, OpBuilder &builder, const Location loc,
    const ArrayRef<int64_t> shape, const std::array<int64_t, 2> dst_tiling,
    const LayoutOffsets dst_offsets_hint, const xla::Array<Value> &src_vregs,
    const VectorLayout &src) {
  const int bitwidth = src.bitwidth();
  const int packing = src.packing();
  const std::array<int64_t, 2> src_tiling = src.tiling();
  if (!(src_tiling[1] == ctx.target_shape[1] &&
        dst_tiling[1] == ctx.target_shape[1] && src_tiling[0] % packing == 0 &&
        dst_tiling[0] % packing == 0)) {
    return failure();
  }
  const std::array<int64_t, 2> src_vreg_slice =
      VectorLayout::vregSlice(ctx.target_shape, bitwidth, src_tiling);
  const std::array<int64_t, 2> dst_vreg_slice =
      VectorLayout::vregSlice(ctx.target_shape, bitwidth, dst_tiling);

  // TODO(b/368088671): When sublane tiling changes, we should be able to
  // preserve some replications from the source layout. But we need to
  // make sure they are implemented efficiently and well-tested. For now, we
  // just simply use 0 for the replicated offset after retiling.
  const LayoutOffsets src_offsets = {src.offsets()[0].value_or(0),
                                     src.offsets()[1].value_or(0)};
  // The provided offset hints are used only if they align with the source
  // offsets, else we default to the smallest possible aligned offsets.
  LayoutOffsets dst_offsets = {*src_offsets[0] % dst_vreg_slice[0],
                               *src_offsets[1] % dst_vreg_slice[1]};
  // On a given dimension, either the source vreg slice size divides the dest
  // vreg slice size, or vice versa (depending on the dimension and whether it's
  // small-to-large or large-to-small retiling). Offset changes are supported
  // as long as they are aligned modulo the smaller of the two sizes.
  const std::array<int64_t, 2> alignment = {
      std::min(src_vreg_slice[0], dst_vreg_slice[0]),
      std::min(src_vreg_slice[1], dst_vreg_slice[1])};
  if (dst_offsets_hint[0].has_value() &&
      (*dst_offsets_hint[0] - *src_offsets[0]) % alignment[0] == 0) {
    CHECK_LT(*dst_offsets_hint[0], dst_vreg_slice[0]);
    dst_offsets[0] = *dst_offsets_hint[0];
  }
  if (dst_offsets_hint[1].has_value() &&
      (*dst_offsets_hint[1] - *src_offsets[1]) % alignment[1] == 0) {
    CHECK_LT(*dst_offsets_hint[1], dst_vreg_slice[1]);
    dst_offsets[1] = *dst_offsets_hint[1];
  }
  // The offsets of the source in units of the destination vreg slice:
  const std::array<int64_t, 2> src_offsets_in_dst_vreg_slices = {
      *src_offsets[0] / dst_vreg_slice[0], *src_offsets[1] / dst_vreg_slice[1]};
  // The offsets of the destination in units of the source vreg slice:
  const std::array<int64_t, 2> dst_offsets_in_src_vreg_slices = {
      *dst_offsets[0] / src_vreg_slice[0], *dst_offsets[1] / src_vreg_slice[1]};

  // Try to get i32 vector scratch space. Because we will bitcast vregs to
  // i32 vregs before using scratch for retiling. Through this way we can
  // handle packed types as well.
  auto vi32_scratch_ref = getInternalScratch(
      ctx, builder, loc, {ctx.max_sublanes_in_scratch, ctx.target_shape[1]},
      builder.getI32Type(), /*sublane_tiling=*/1);
  if (failed(vi32_scratch_ref)) {
    return emitError(loc, "Failed to get scratch ref for retiling");
  }
  auto ref = vi32_scratch_ref.value();
  std::array<int64_t, 2> vi32_dst_tiling = {dst_tiling[0] / packing,
                                            dst_tiling[1]};
  std::array<int64_t, 2> vi32_src_tiling = {src_tiling[0] / packing,
                                            src_tiling[1]};

  const VectorLayout dst(bitwidth, dst_offsets, dst_tiling, src.implicit_dim());
  TPU_ASSERT_LOC(loc, dst.isValid(ctx.target_shape));
  xla::Array<Value> dst_vregs(
      dst.tileArrayImplicitShape(shape, ctx.target_shape));
  // When differences in offsets exist, the source vregs may stored at an offset
  // position in their group. For example, the 1st vreg in a row/column may be
  // stored as if it was the 3rd, so that the parts corresponding to the 1st and
  // 2nd in the destination are filled with padding. Likewise, loads to
  // destination vregs may be skipped, when they would load only padding.
  // store_vreg_delay is the position offset for stores, and load_vreg_skips is
  // the position offset for loads.
  //
  // For example, suppose we are going from 32-bit {0, 128}(2, 128) to
  // {4, 0}(8, 128). We form groups of 4 vregs that represent an (8, 512) slice
  // of the padded implicit shape. For the given offsets, for the first group,
  // the data is in (4:8, 128:512). But the first and second sources (stored
  // vregs) of the group form the slices of data (0:2, 0:512) and (2:4, 0:512),
  // which should be all padding. Likewise, the first dest vreg slice (which we
  // load from) holds the data from slice (0:8, 0:128), which is all padding.
  // We never load or store to slices that should contain only padding.
  if (src_tiling[0] > dst_tiling[0]) {
    DCHECK_EQ(src_offsets_in_dst_vreg_slices[1], 0);
    DCHECK_EQ(dst_offsets_in_src_vreg_slices[0], 0);
    const int64_t store_vreg_delay = dst_offsets_in_src_vreg_slices[1];
    const int64_t load_vreg_skips = src_offsets_in_dst_vreg_slices[0];
    if (failed(retileToSmallTileWithScratch(
            ctx, builder, loc, dst_vregs, vi32_dst_tiling, src_vregs,
            vi32_src_tiling, ref, store_vreg_delay, load_vreg_skips))) {
      return failure();
    }
  }
  if (src_tiling[0] < dst_tiling[0]) {
    DCHECK_EQ(src_offsets_in_dst_vreg_slices[0], 0);
    DCHECK_EQ(dst_offsets_in_src_vreg_slices[1], 0);
    const int64_t store_vreg_delay = dst_offsets_in_src_vreg_slices[0];
    const int64_t load_vreg_skips = src_offsets_in_dst_vreg_slices[1];
    if (failed(retileToLargeTileWithScratch(
            ctx, builder, loc, dst_vregs, vi32_dst_tiling, src_vregs,
            vi32_src_tiling, ref, store_vreg_delay, load_vreg_skips))) {
      return failure();
    }
  }
  return std::make_pair(dst, dst_vregs);
}

FailureOr<std::pair<VectorLayout, xla::Array<Value>>> changeTiling(
    RewriteContext &ctx, OpBuilder &builder, const Location loc, VectorType vty,
    const VectorLayout src, xla::Array<Value> vregs,
    const std::array<int64_t, 2> dst_tiling,
    const LayoutOffsets dst_offsets_hint) {
  bool has_enough_scratch = ctx.max_sublanes_in_scratch >=
                            ctx.target_shape[0] * (ctx.target_shape[0] + 1);
  const auto &target_shape = ctx.target_shape;
  const std::array<int64_t, 2> src_tiling = src.tiling();
  if (src_tiling == dst_tiling) {
    return std::pair(src, std::move(vregs));
  }
  const LayoutOffsets src_offsets =
      src.getCanonicalOffsets(vty.getShape(), ctx.target_shape);
  const std::array<int64_t, 2> tiled_ishape =
      src.getImplicitTiledDims(vty.getShape(), 1);
  const int packing = src.packing();
  const int8_t bitwidth = src.bitwidth();
  const std::array<int64_t, 2> dst_vreg_slice =
      VectorLayout::vregSlice(ctx.target_shape, bitwidth, dst_tiling);
  // TODO(tlongeri): Using canonical vs non-canonical offsets can change the
  // value of try_replicate rows, and it breaks some tests. It doesn't make
  // sense that we have different behavior for equivalent layouts, though. We
  // need better logic for picking the relayout strategy.
  const bool try_replicate_rows =
      src.offsets()[0].has_value() && !dst_offsets_hint[0].has_value();

  // Fully replicated offsets are handled efficiently elsewhere (in relayout)
  CHECK(src.offsets()[0].has_value() || src.offsets()[1].has_value());

  // Handle replicating small-to-large retiling for (a) replicated 2nd minor or
  // (b) 32-bit single-row.
  // This retiling is one-to-many vregs.
  // TODO(tlongeri): Large-to-small retiling with replicated minor is analogous
  // to this.
  if (src_tiling[1] == ctx.target_shape[1] &&
      dst_tiling[1] == ctx.target_shape[1] &&
      dst_tiling[0] % src_tiling[0] == 0 &&
      (!src_offsets[0].has_value() || (packing == 1 && tiled_ishape[0] == 1)) &&
      // This relayout relies on gathers, which are cheap on newer generations,
      // so we always use it for them.
      // TODO(tlongeri): Once we have it, probably also prefer the
      // small-to-large rotate+blend relayout if we don't need replication. It's
      // slightly cheaper for some dst vregs you rotate by 0.
      // TODO(tlongeri): Using store + multiple replicated loads is good on
      // older gens. I wonder if we can integrate this logic to scratch retiling
      (try_replicate_rows || ctx.hardware_generation >= 5)) {
    const LayoutOffset dst_minor_offset =
        src.offsets()[1].has_value() ? *src.offsets()[1] % dst_vreg_slice[1]
                                     : LayoutOffset();
    const VectorLayout dst(bitwidth, {std::nullopt, dst_minor_offset},
                           dst_tiling, src.implicit_dim());
    const SmallVector<int64_t> dst_vreg_array_shape =
        dst.tileArrayImplicitShape(vty.getShape(), target_shape);
    const int64_t src_tiles_per_vreg = src.tilesPerVreg(ctx.target_shape);
    const int64_t dst_tiles_per_vreg = dst.tilesPerVreg(ctx.target_shape);
    const int64_t src_sublanes_per_tile = src.sublanesPerTile(ctx.target_shape);
    const int64_t dst_sublanes_per_tile = dst.sublanesPerTile(ctx.target_shape);
    xla::Array<Value> retiled(dst_vreg_array_shape);
    SmallVector<int64_t> idxs;
    retiled.Each([&](absl::Span<const int64_t> dst_idx, Value *vreg) {
      const int64_t dst_col_idx = *(dst_idx.end() - 1);
      const int64_t base_dst_tile_idx = dst_col_idx * dst_tiles_per_vreg;
      const int64_t base_src_tile_idx =
          src_offsets[1].has_value()
              ? base_dst_tile_idx +
                    (*src_offsets[1] - *dst_minor_offset) / src_tiling[1]
              : 0;
      // The following should be true from our choice of minor offset:
      DCHECK_EQ(base_src_tile_idx % dst_tiles_per_vreg, 0);
      const int64_t src_col_idx = base_src_tile_idx / src_tiles_per_vreg;
      SmallVector<int32_t, 8> gather_pattern;
      // Iterate over the sublanes in the dst vreg:
      for (int32_t sublane = 0; sublane < ctx.target_shape[0]; ++sublane) {
        const int64_t dst_tile_idx_in_vreg = sublane / dst_sublanes_per_tile;
        const int64_t src_tile_idx_in_vreg =
            base_src_tile_idx % src_tiles_per_vreg + dst_tile_idx_in_vreg;
        // Although replication may give us several sublanes to choose from,
        // we always gather from the first sublane in the source tile. This
        // degenerates to a broadcast when dst_tiling is native, which can
        // be cheaper than an arbitrary gather (for some hardware gens).
        const int64_t src_sublane_in_tile =
            src_offsets[0].value_or(0) / packing;
        const int64_t src_sublane =
            src_tile_idx_in_vreg * src_sublanes_per_tile + src_sublane_in_tile;
        gather_pattern.push_back(src_sublane);
      }
      idxs.assign(dst_idx.begin(), dst_idx.end());
      *(idxs.end() - 2) = 0;
      *(idxs.end() - 1) = src_col_idx;
      Value src_vreg = vregs(idxs);
      *vreg = builder.create<tpu::GatherOp>(loc, src_vreg.getType(), src_vreg,
                                            gather_pattern,
                                            /*dimension=*/0);
    });
    return std::pair(dst, std::move(retiled));
  }
  // (8,128) -> (8 * packing,128) tiling change for packed type.
  if (src_offsets[0].value_or(0) < dst_vreg_slice[0] &&
      src_offsets[1].value_or(0) < dst_vreg_slice[1] && bitwidth < 32 &&
      32 % bitwidth == 0 && src_tiling == ctx.target_shape &&
      dst_tiling == std::array<int64_t, 2>{ctx.target_shape[0] * packing,
                                           ctx.target_shape[1]}) {
    // TODO(tlongeri): This relayout is just ext + trunc. Refactor.
    // Note: for int4, retiling with scratch is always faster.
    if (bitwidth != 4 || !has_enough_scratch) {
      // Note: The code below does not work when src is replicated and dst is
      // not, since it relies on the src vreg array shape to know how many tiles
      // to pack in dst, and vreg array shapes with materialized offsets are
      // unfortunately not equal to vreg array shapes with replicated offsets.
      VectorLayout dst(src.bitwidth(), src.offsets(), dst_tiling,
                       src.implicit_dim());
      xla::Array<Value> retiled(
          dst.tileArrayImplicitShape(vty.getShape(), target_shape));
      VectorType vreg_x32 =
          vty.getElementType().isSignlessInteger()
              ? VectorType::get(target_shape, builder.getI32Type())
              : VectorType::get(target_shape, builder.getF32Type());
      retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
        const int vreg_part = idx.back() % packing;
        SmallVector<Value, 8> parts;
        parts.reserve(packing);
        SmallVector<int64_t> src_idx(idx.begin(), idx.end());
        *(src_idx.end() - 1) /= packing;
        if (!dst.offsets()[0].has_value()) {
          *(src_idx.end() - 2) = 0;
          // Make sure we set all parts of the output vreg to make it replicated
          parts.append(packing, builder.create<tpu::UnpackSubelementsOp>(
                                    loc, vreg_x32, vregs(src_idx), vreg_part,
                                    tpu::PackFormat::kCompressed));
        } else {
          *(src_idx.end() - 2) *= packing;
          for (int i = 0; i < packing; ++i) {
            if (*(src_idx.end() - 2) < *(vregs.dimensions().end() - 2)) {
              parts.push_back(builder.create<tpu::UnpackSubelementsOp>(
                  loc, vreg_x32, vregs(src_idx), vreg_part,
                  tpu::PackFormat::kCompressed));
              ++*(src_idx.end() - 2);
            } else {
              parts.push_back(nullptr);
            }
          }
        }
        *tile = builder.create<tpu::PackSubelementsOp>(
            loc, cast<VectorType>(vregs.begin()->getType()), parts,
            tpu::PackFormat::kCompressed);
      });
      return std::pair(dst, std::move(retiled));
    }
  }
  // Handle retiling from (1, 128 * packing) to (packing, 128) for
  // packed data.
  // We do compressed unpacking followed by interleaved packing.
  // TODO(tlongeri): This can be used as a first step before using
  // a generalized retiling where we only move sublanes around
  // (without packing/unpacking).
  // TODO(tlongeri): Interleaved unpacking followed by interleaved
  // packing (but with different pairings) might also be
  // interesting if the next step is a retile, since we can also
  // match corresponding elements without shifting. It's just that
  // the tiles are not adjacent (no contiguous vreg slice).
  if (src_offsets[0].value_or(0) < dst_vreg_slice[0] &&
      src_offsets[1].value_or(0) < dst_vreg_slice[1] && bitwidth < 32 &&
      32 % bitwidth == 0 &&
      src_tiling == std::array<int64_t, 2>{1, ctx.target_shape[1] * packing} &&
      dst_tiling == std::array<int64_t, 2>{packing, ctx.target_shape[1]}) {
    // TODO(tlongeri): This relayout is just ext + trunc. Refactor.
    // To illustrate, consider a 2 x 16 16-bit shape laid out in vregs of
    // 4 sublanes and 2 lanes (this is convenient for to keep the example small
    // yet non-trivial) with (1, 4) tiling. We will relayout to (2, 2) tiling.
    //
    // The vreg slice is 1 x 16, that is, the vreg contains the data for a
    // 1 x 16 window of the logical shape.
    //
    // [a b c d e f g h i j k l m n o p] -> vreg 1
    // [A B C D E F G H I J K L M N O P] -> vreg 2
    //
    // Note: we support multiple vregs per row of the logical shape, but we use
    //       one here just to keep the example small.
    //
    // When we do a compressed unpack, the resulting vregs effectively have a
    // tiling of (1, 2) and cover a vreg slice of 1 x 8 logical elements.
    //
    // [a b c d e f g h] -> vreg 1, part 1   [i j k l m n o p] -> vreg 1, part 2
    // [A B C D E F G H] -> vreg 2, part 1   [I J K L M N O P] -> vreg 2, part 2
    //
    // It is clear that if combine vreg 1, part 1 and vreg 2, part 1 we get data
    // that covers a 2 x 8 vreg slice. Note, however, that we will have to mind
    // the internal ordering of the vreg.
    //
    // [a b c d e f g h                      [i j k l m n o p
    //  A B C D E F G H] -> new vreg 1        I J K L M N O P] -> new vreg 2
    //
    // To see if we can get the right internal ordering that we need for (2, 2)
    // tiling, let's break new vreg 1 into (1, 2) rows, which correspond to
    // sublanes when unpacked and half-sublanes when packed.
    //
    // [(a b) (c d) (e f) (g h)
    //  (A B) (C D) (E F) (G H)]
    //
    // The sublane order for the vreg parts is [(a b) (c d) ...] for vreg 1,
    // part 1 and [(A B) (C D) ...] for vreg 2, part 1.
    //
    // The desired half-sublane order, for packed (2, 2) tiling, is
    // [(a b) (A B) (c d) (C D) ...]. That is, traverse down each column before
    // moving to the next one. This is exactly an interleaving of the sublanes
    // of the vreg parts.

    // Note: The code below does not work when src is replicated and dst is
    // not, since it relies on the src vreg array shape to know how many tiles
    // to pack in dst, and vreg array shapes with materialized offsets are
    // unfortunately not equal to vreg array shapes with replicated offsets.
    VectorLayout dst(src.bitwidth(), src.offsets(), dst_tiling,
                     src.implicit_dim());
    xla::Array<Value> retiled(
        dst.tileArrayImplicitShape(vty.getShape(), target_shape));
    const VectorType vreg_x32 =
        vty.getElementType().isSignlessInteger()
            ? VectorType::get(target_shape, builder.getI32Type())
            : VectorType::get(target_shape, builder.getF32Type());
    retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      SmallVector<Value> parts;
      parts.reserve(packing);
      SmallVector<int64_t> src_idx(toArrayRef(idx));
      const int64_t vreg_part = *(src_idx.end() - 1) % packing;
      *(src_idx.end() - 1) /= packing;
      if (!dst.offsets()[0].has_value()) {
        *(src_idx.end() - 2) = 0;
        // Make sure we set all parts of the output vreg to make it replicated
        parts.append(packing, builder.create<tpu::UnpackSubelementsOp>(
                                  loc, vreg_x32, vregs(src_idx), vreg_part,
                                  tpu::PackFormat::kCompressed));
      } else {
        *(src_idx.end() - 2) *= packing;
        for (int i = 0; i < packing; ++i) {
          if (*(src_idx.end() - 2) < *(vregs.dimensions().end() - 2)) {
            parts.push_back(builder.create<tpu::UnpackSubelementsOp>(
                loc, vreg_x32, vregs(src_idx), vreg_part,
                tpu::PackFormat::kCompressed));
            ++*(src_idx.end() - 2);
          } else {
            parts.push_back(nullptr);
          }
        }
      }
      *tile = builder.create<tpu::PackSubelementsOp>(
          loc, cast<VectorType>(vregs.begin()->getType()), parts,
          tpu::PackFormat::kInterleaved);
    });
    return std::pair(dst, std::move(retiled));
  }
  if (src_tiling[1] == target_shape[1] && dst_tiling[1] == target_shape[1]) {
    // All clauses in the and expression are based on performance benchmarking.
    bool use_alu = !has_enough_scratch ||
                   (ctx.hardware_generation >= 5 && src_tiling[0] != packing &&
                    dst_tiling[0] != packing);

    if (use_alu) {
      if (src_tiling[0] > dst_tiling[0] &&
          // retileToReducedSublanes does not support offset changes
          src.offsets()[0].value_or(0) < dst_vreg_slice[0] &&
          src.offsets()[1].value_or(0) < dst_vreg_slice[1]) {
        VectorLayout dst(src.bitwidth(), src.offsets(), dst_tiling,
                         src.implicit_dim());
        return std::pair(dst, retileToReducedSublanes(
                                  builder, vty.getShape(), src, vregs,
                                  VectorLayout(bitwidth,
                                               {src.offsets()[0].value_or(0),
                                                src.offsets()[1].value_or(0)},
                                               dst_tiling, dst.implicit_dim()),
                                  target_shape));
      } else if (!has_enough_scratch) {
        // TODO(b/357538782): Implement retileToIncreasedSublanes with ALU ops.
        return emitError(
            loc,
            "Not implemented: retiling to increase sublane tiling with ALU");
      }
    }
    return retileWithScratch(ctx, builder, loc, vty.getShape(), dst_tiling,
                             dst_offsets_hint, vregs, src);
  }
  return emitError(loc, "Not implemented: Unsupported tiling change for ")
         << vty << ": from " << src << " to (" << dst_tiling[0] << ", "
         << dst_tiling[1] << ") tiling";
}

FailureOr<std::pair<VectorLayout, xla::Array<Value>>> changeImplicitDim(
    RewriteContext &ctx, OpBuilder &builder, const Location loc, VectorType vty,
    const VectorLayout src, xla::Array<Value> vregs,
    const VectorLayout::ImplicitDim dst_implicit_dim,
    const LayoutOffsets dst_offset_hints) {
  const auto &target_shape = ctx.target_shape;
  if (src.implicit_dim() == dst_implicit_dim) {
    return std::make_pair(src, std::move(vregs));
  }
  // It's possible that the implicit dim change is a no-op.
  VectorLayout src_candidate(src.bitwidth(), src.offsets(), src.tiling(),
                             dst_implicit_dim);
  if (src_candidate.equivalentTo(src, vty.getShape(), target_shape)) {
    vregs.Reshape(
        src_candidate.tileArrayImplicitShape(vty.getShape(), target_shape));
    return std::make_pair(src_candidate, vregs);
  }
  // Remove second minor implicit dim, for values that have (m, 128) tiling (for
  // m that is a power of 2).
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kSecondMinor &&
      dst_implicit_dim == VectorLayout::ImplicitDim::kNone &&
      src.bitwidth() == 32 && src.tiling()[1] == target_shape[1] &&
      llvm::isPowerOf2_32(src.tiling()[0])) {
    // We should never see a replicated offset here. We're removing the implicit
    // dim so the only case when this can happen is when its size is 1 (or else
    // we can't prove replication in the logical value). But in that case, the
    // equivalentTo case above triggers and we never reach this branch.
    CHECK(dst_offset_hints[0].has_value());
    int64_t dst_sublane_offset = *dst_offset_hints[0];
    VectorLayout dst(src.bitwidth(), {dst_sublane_offset, src.offsets()[1]},
                     src.tiling(), dst_implicit_dim);
    xla::Array<Value> new_vregs(
        dst.tileArrayImplicitShape(vty.getShape(), target_shape));
    new_vregs.Each([&](const absl::Span<const int64_t> idx, Value *tile) {
      const int64_t dst_2nd_minor_idx = idx.size() - 2;
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      src.insertImplicit<int64_t>(src_idx, 0);
      const int dst_sl_start =
          idx[dst_2nd_minor_idx] == 0 ? dst_sublane_offset : 0;
      // This could be optimized further to take offsets[1] into account.
      // For example, extended offsets allow us to skip copies of low sublanes
      // in tiles with idx.back() == 0.
      const int tiles_per_vreg = src.tilesPerVreg(target_shape);
      const int sublanes_per_tile = src.sublanesPerTile(target_shape);
      src_idx[dst_2nd_minor_idx] = src.tiling()[0] * idx[dst_2nd_minor_idx] +
                                   dst_sl_start - dst_sublane_offset;
      for (int dst_sl_idx = dst_sl_start;
           dst_sl_idx < src.tiling()[0] &&
           src_idx[dst_2nd_minor_idx] < vregs.dim(dst_2nd_minor_idx);
           ++dst_sl_idx, ++src_idx[dst_2nd_minor_idx]) {
        // This could be optimized further by copying multiple sublanes at once.
        for (int tile_idx = 0; tile_idx < tiles_per_vreg; ++tile_idx) {
          int tile_off = tile_idx * sublanes_per_tile;
          *tile =
              copy_one_sublane(builder, vregs(src_idx),
                               tile_off + src.offsets()[0].value_or(dst_sl_idx),
                               *tile, tile_off + dst_sl_idx, target_shape);
        }
      }
    });
    return std::make_pair(dst, new_vregs);
  }
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst_implicit_dim == VectorLayout::ImplicitDim::kMinor &&
      src.bitwidth() == 32 && src.hasNativeTiling(ctx.target_shape)) {
    // TODO(tlongeri): Make insertImplicitMinorDimension more flexible about
    //                 offsets, then we can pass dst_offset_hints directly.
    const LayoutOffset dst_2nd_minor_offset =
        !src.offsets()[1] || *src.offsets()[1] + *(vty.getShape().end() - 1) <=
                                 ctx.target_shape[1]
            ? dst_offset_hints[0]
            : LayoutOffset(*src.offsets()[1] % ctx.target_shape[0]);
    VectorLayout dst(src.bitwidth(),
                     {dst_2nd_minor_offset, dst_offset_hints[1]}, src.tiling(),
                     VectorLayout::ImplicitDim::kMinor);
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> dst_vregs,
        insertImplicitMinorDimension(ctx, builder, loc, vregs,
                                     src.implicitShape(vty.getShape()), src,
                                     dst.offsets()));
    return std::make_pair(dst, std::move(dst_vregs));
  }
  return emitError(loc,
                   "Not implemented: Unsupported implicit dim change: from ")
         << src << " to " << dst_implicit_dim;
}

// TODO(apaszke): Test this function properly
FailureOr<TypedValue<VectorType>> relayout(RewriteContext &ctx,
                                           OpBuilder &builder,
                                           TypedValue<VectorType> v,
                                           VectorLayout src,
                                           VectorLayout dst) {
  const auto target_shape = ctx.target_shape;
  const int8_t bitwidth = src.bitwidth();
  if (bitwidth != dst.bitwidth()) {
    return emitError(v.getLoc(), "Can't change bitwidth during a relayout");
  }
  VectorType vty = v.getType();
  const bool is_mask = vty.getElementTypeBitWidth() == 1;
  if (is_mask) {
    if (src.bitwidth() != 32 || dst.bitwidth() != 32) {
      return emitError(v.getLoc(),
                       "Not implemented: mask relayout with non-32 bitwidth in "
                       "vector layout");
    }
  }
  {
    // Replication imposes a replication constraint on the *logical* value of
    // the vector: When moving along a replicated axis, all elements must be
    // equal. Note that when the axis is a singleton, there is effectively no
    // added *logical* constraint.
    // For example, a vector<2x2xf32> v with no implicit dims and layout offsets
    // {*, 0} is expected to satisfy v[0, 0] == v[1, 0] and v[0, 1] == v[1, 1].
    // Relayout does not change the logical value of the vector. Any replication
    // constraints in the result must be guaranteed by the source layout.
    SmallVector<LayoutOffset, 2> src_offsets(ArrayRef(src.offsets()));
    SmallVector<LayoutOffset, 2> dst_offsets(ArrayRef(dst.offsets()));
    // Remove implicit dims to get offsets for trailing logical dims.
    src.eraseImplicit(src_offsets);
    dst.eraseImplicit(dst_offsets);
    for (int i = dst_offsets.size(); i > 0; --i) {
      const int64_t dim_size = *(vty.getShape().end() - i);
      const bool dim_replicated_in_dst = !*(dst_offsets.end() - i);
      // If the dim is untiled in the src layout, then there is no guarantee of
      // replication, because we don't track replication for untiled dims.
      const bool dim_replicated_in_src =
          i <= src_offsets.size() && !*(src_offsets.end() - i);
      if (dim_replicated_in_dst && !dim_replicated_in_src && dim_size != 1) {
        return emitError(v.getLoc(),
                         "Invalid relayout: Non-singleton logical dimension is "
                         "replicated in destination but not in source for ")
               << vty << ": " << src << " -> " << dst;
      }
    }
  }

  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_tiles,
      disassemble(builder, src, v, target_shape, /*use_implicit_shape=*/true));
  if (is_mask) {
    auto new_tile_ty =
        getNativeVregOrVmaskType(builder.getI32Type(), 32, target_shape);
    src_tiles.Each([&](const absl::Span<const int64_t> idx, Value *tile) {
      *tile =
          builder.create<arith::ExtUIOp>(tile->getLoc(), new_tile_ty, *tile);
    });
    vty = VectorType::get(vty.getShape(), builder.getI32Type());
  }
  auto assemble_with_mask_check = [&](xla::Array<Value> &tiles,
                                      bool use_implicit_shape = false) {
    if (is_mask) {
      auto zeros_tile = builder.create<arith::ConstantOp>(
          tiles.begin()->getLoc(),
          DenseElementsAttr::get(cast<VectorType>(tiles.begin()->getType()),
                                 builder.getI32IntegerAttr(0)));
      tiles.Each([&](const absl::Span<const int64_t> idx, Value *tile) {
        *tile = builder.create<arith::CmpIOp>(
            tile->getLoc(), arith::CmpIPredicate::ne, *tile, zeros_tile);
      });
      vty = VectorType::get(vty.getShape(), builder.getI1Type());
    }
    return assemble(builder, vty, dst, tiles, target_shape, use_implicit_shape)
        .getResult();
  };
  // Two easy cases: source is more general, or is replicated.
  if (src.generalizes(dst, vty.getShape(), target_shape)) {
    // A value with a replicated offset might use fewer vregs than a value with
    // a non-zero offset.
    auto src_product =
        xla::Product(src.tileArrayShape(vty.getShape(), target_shape));
    auto dst_product =
        xla::Product(dst.tileArrayShape(vty.getShape(), target_shape));
    if (src_product != dst_product) {
      TPU_ASSERT_LOC(v.getLoc(), dst_product > src_product);
      auto src_offsets = src.offsets();

      TPU_ASSERT_LOC(v.getLoc(), src_offsets != dst.offsets());
      TPU_ASSERT_LOC(v.getLoc(), src.bitwidth() == dst.bitwidth());

      if (src.implicit_dim() != dst.implicit_dim()) {
        return emitError(v.getLoc(),
                         "Not implemented: Source layout is more general, but "
                         "vreg count changes and implicit dims are mismatched");
      }

      if (src.tiling() != dst.tiling()) {
        return emitError(v.getLoc(),
                         "Not implemented: Source layout is more general, but "
                         "vreg count changes and tiling are mismatched");
      }

      // This case is moving from a replicated to a non replicated layout.
      // As such, we need to make a new destination shape that is the
      // materialization of the src shape with replication.
      FAILUREOR_ASSIGN_OR_RETURN(auto src_vregs,
                                 disassemble(builder, src, v, target_shape,
                                             /*use_implicit_shape=*/true));
      auto dst_vregs_shape = dst.tileArrayShape(vty.getShape(), target_shape);
      xla::Array<Value> dst_vregs(dst_vregs_shape);
      dst_vregs.Each([&](const absl::Span<const int64_t> idx, Value *vreg) {
        SmallVector<int64_t> local_idx(idx.begin(), idx.end());
        if (!src_offsets[0].has_value()) {
          local_idx[local_idx.size() - 2] = 0;
        }
        if (!src_offsets[1].has_value()) {
          local_idx[local_idx.size() - 1] = 0;
        }
        *vreg = src_vregs(local_idx);
      });
      return assemble(builder, vty, dst, std::move(dst_vregs), target_shape,
                      /*use_implicit_shape=*/true)
          .getResult();
    }
    src_tiles.Reshape(dst.tileArrayImplicitShape(vty.getShape(), target_shape));
    return assemble_with_mask_check(src_tiles,
                                    /*use_implicit_shape=*/true);
  }

  if (const LayoutOffsets src_offsets =
          src.getCanonicalOffsets(vty.getShape(), ctx.target_shape);
      src.layout_rank() >= dst.layout_rank() && !src_offsets[0].has_value() &&
      !src_offsets[1].has_value()) {
    // A fully replicated value is always easy to relayout
    xla::Array<Value> dst_tiles(
        dst.tileArrayImplicitShape(vty.getShape(), target_shape));
    SmallVector<int64_t> idxs;
    dst_tiles.Each([&](const absl::Span<const int64_t> src_idx, Value *vreg) {
      idxs.assign(src_idx.begin(), src_idx.end());
      dst.eraseImplicit(idxs);
      src.insertImplicit<int64_t>(idxs, 0);
      *(idxs.end() - 2) = 0;
      *(idxs.end() - 1) = 0;
      *vreg = src_tiles(idxs);
    });
    return assemble_with_mask_check(dst_tiles, /*use_implicit_shape=*/true);
  }

  // Consider (1,128),-2 -> (8,128). In this case we can change the implicit
  // dim for free before we change the tiling, but not after.
  // TODO(apaszke): In general the number of vregs necessary to represent a
  // value for different implicit dims satisfies kNone < kSecondMinor < kMinor.
  // We should use this property to decide if we should change the implicit dim
  // before or after changing the tiling and offsets.
  if (src.implicit_dim() != dst.implicit_dim()) {
    VectorLayout src_candidate(src.bitwidth(), src.offsets(), src.tiling(),
                               dst.implicit_dim());
    if (src_candidate.equivalentTo(src, vty.getShape(), target_shape)) {
      src = src_candidate;
      src_tiles.Reshape(
          src.tileArrayImplicitShape(vty.getShape(), target_shape));
    }
  }

  FAILUREOR_ASSIGN_OR_RETURN(
      std::tie(src, src_tiles),
      changeTiling(ctx, builder, v.getLoc(), vty, src, std::move(src_tiles),
                   dst.tiling(), dst.offsets()));

  FAILUREOR_ASSIGN_OR_RETURN(
      std::tie(src, src_tiles),
      changeImplicitDim(ctx, builder, v.getLoc(), vty, src,
                        std::move(src_tiles), dst.implicit_dim(),
                        dst.offsets()));

  FAILUREOR_ASSIGN_OR_RETURN(
      std::tie(src, src_tiles),
      changeOffsets(ctx, builder, v.getLoc(), vty, src, std::move(src_tiles),
                    dst.offsets()));

  CHECK_EQ(src, dst);  // At this point we've should be done.
  return assemble_with_mask_check(src_tiles,
                                  /*use_implicit_shape=*/true);
}

// TODO(apaszke): Implement a debug mode that inserts additional assertions.
// For example, we should verify that ops that were supposed to generate
// replicated outputs satisfy that requirement.
LogicalResult applyLayoutOp(RewriteContext &ctx, Operation &op) {
  // When an operation does not have any operands, the layout_in tuple is empty.
  // If one of the operands is not of vector type, the corresponding entry in
  // the layout_in tuple will be None. The same applies to the results of the
  // operation and the layout_out tuple.
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layouts_out,
                             getOutLayouts(op, ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layouts_in,
                             getInLayouts(op, ctx.target_shape));
  if (!layouts_in.empty() && !isa<tpu::AssumeLayoutOp>(op)) {
    // Relayout the operands, if their requested input layouts don't match the
    // layouts in which they were produced.
    for (auto [idx, tup] :
         llvm::enumerate(llvm::zip(op.getOperands(), layouts_in))) {
      auto [operand, li] = tup;
      auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand);
      TPU_ASSERT_EQ_OP(vector_operand != nullptr, li.has_value());
      if (vector_operand == nullptr) {
        continue;
      }
      // The operand should always be an Operation (and not a BlockArgument)
      // since we expect the FuncOp to have only memrefs and semaphores as
      // arguments.
      auto op_result = dyn_cast<OpResult>(vector_operand);
      if (op_result == nullptr) {
        return op.emitError(
            "Expected vector operand to be an operation result");
      }
      Operation *const def_op = op_result.getOwner();
      DCHECK(def_op);
      const unsigned res_idx = op_result.getResultNumber();
      FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                                 getOutLayouts(*def_op, ctx.target_shape));
      const Layout lo = def_layouts[res_idx];
      TPU_ASSERT_OP(lo.has_value());
      if (*lo == *li) {
        continue;
      }
      OpBuilder builder(&op);
      FAILUREOR_ASSIGN_OR_RETURN(
          Value new_v, relayout(ctx, builder, vector_operand, /*src=*/*lo,
                                /*dst=*/*li));
      op.setOperand(idx, new_v);
    }
  }

  // TODO: b/342235360 - This check is temporary while we increase and test
  // support for offsets outside of the first tile. When support is more broad,
  // any op without support should check it within their own rule.
  if (!isa<arith::TruncFOp, arith::TruncIOp, vector::BroadcastOp,
           vector::ExtractStridedSliceOp, vector::ShapeCastOp>(op)) {
    for (const Layout &layout : layouts_in) {
      if (layout && layout->offsets()[1].has_value() &&
          layout->offsets()[1].value() >= layout->tiling()[1]) {
        return op.emitError(
            "Not implemented: Input offsets outside of the first tile");
      }
    }
  }
  const bool no_vector_args =
      llvm::none_of(layouts_out,
                    [](Layout layout) { return layout.has_value(); }) &&
      llvm::none_of(layouts_in,
                    [](Layout layout) { return layout.has_value(); });
  if (no_vector_args && op.getRegions().empty()) {
    // We don't need to do anything for scalar operations.
    if (!op.getOperands().empty()) {
      op.removeAttr("in_layout");
    }
    if (!op.getResults().empty()) {
      op.removeAttr("out_layout");
    }
    return success();
  }
  if (auto rule_it = rules().find(op.getName().getStringRef());
      rule_it != rules().end()) {
    const rule_type &rule = rule_it->getValue();
    return rule(ctx, op, layouts_in, layouts_out);
  }
  if (OpTrait::hasElementwiseMappableTraits(&op)) {
    return elementwise_op_rule(ctx, op, layouts_in, layouts_out);
  }
  return op.emitError("Not implemented: Unsupported operation: ")
         << op.getName();
}

LogicalResult applyLayoutBlock(RewriteContext &ctx, Block &block) {
  // We'll be modifying the block, so use early increment.
  for (Operation &op : make_early_inc_range(block)) {
    if (failed(applyLayoutOp(ctx, op))) {
      return failure();
    }
  }
  return success();
}

// Rewrites the function according to layout annotations of its operations.
//
//   Args:
//     ctx: The context used for rewriting.
//     f: An MLIR function to be rewritten.
LogicalResult applyLayoutFunc(RewriteContext &ctx, func::FuncOp f) {
  if (f->getNumRegions() != 1) {
    return f.emitError("Expected FuncOp to have a single region");
  }
  if (!f.getBody().hasOneBlock()) {
    return f.emitError("Expected FuncOp to have a single block");
  }
  return applyLayoutBlock(ctx, f.getBody().front());
}

struct ApplyVectorLayoutPass
    : public impl::ApplyVectorLayoutPassBase<ApplyVectorLayoutPass> {
  ApplyVectorLayoutPass(const RewriteContext &ctx) {
    hardware_generation = ctx.hardware_generation;
    sublane_count = ctx.target_shape[0];
    lane_count = ctx.target_shape[1];
    mxu_contracting_size = ctx.mxu_shape[0];
    mxu_noncontracting_size = ctx.mxu_shape[1];
    max_sublanes_in_scratch = ctx.max_sublanes_in_scratch;
    vmem_banks = ctx.vmem_banks;
    max_shuffle_sublane_offset = ctx.max_shuffle_sublane_offset;
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      signalPassFailure();
      return;
    }
    RewriteContext ctx{
        .hardware_generation = hardware_generation,
        .target_shape = {sublane_count, lane_count},
        .mxu_shape = {mxu_contracting_size, mxu_noncontracting_size},
        .max_sublanes_in_scratch = max_sublanes_in_scratch,
        .vmem_banks = vmem_banks,
        .max_shuffle_sublane_offset = max_shuffle_sublane_offset,
    };
    if (failed(applyLayoutFunc(ctx, getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createApplyVectorLayoutPass(
    const RewriteContext &ctx) {
  return std::make_unique<ApplyVectorLayoutPass>(ctx);
}
}  // namespace mlir::tpu