# Author: Oliver Granert
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic example using JAX to convolve a 3d volume image (NIFTI image)

For image processing the example is based on the JAX submodule lax
which provides the convolution function lax.conv_general_dilated.
Reading is done by nibable and final plotting by matplotlib.
"""

# For JAX processing
from jax import lax
import jax.numpy as jnp

# For image import (read/write NIFTI)
import nibabel as nib

# For plotting
import matplotlib.pyplot as plt

# Read 3d image data
img_in = nib.load("ch2bet.nii.gz")

# Transfer image data to jax
JNPIMG = jnp.array( img_in.get_fdata() )

# A 3D kernel in HWDIO layout
# (H)eight, (W)idth, (D)epth, (I)nput channel, (O)utput channel
# Note:
# A 3D kernel is not required for this, but demonstrate the general possibility!
kernel = jnp.array([
  [[0, 0,  0], [0,  -1,  0], [0,  0,   0]],
  [[0, 0, 0], [0, 0, 0], [0,  0,  0]],
  [[0, 0,  0], [0,  1,  0], [0,  0, 0]]],
  dtype=jnp.float32)[:, :, :, jnp.newaxis, jnp.newaxis]

# Reshape input volume
JNPIMG2 = JNPIMG[jnp.newaxis, :, :, :, jnp.newaxis]
print( JNPIMG2.shape )

dn = lax.conv_dimension_numbers(JNPIMG2.shape, kernel.shape,
                                ('NHWDC', 'HWDIO', 'NHWDC'))
print(dn)

# Convolve data with specified 3d kernel
out = lax.conv_general_dilated(
               JNPIMG2,   # lhs = NCHW image tensor
               kernel,    # rhs = OIHW conv kernel tensor
               (1, 1, 1), # window strides
               'SAME',
               (1,1,1),   # lhs/image dilation
               (1,1,1),   # rhs/kernel dilation
               dn)        # dimension_numbers = lhs, rhs, out dimension permutation

# Plot a slice from input and result volume
fig = plt.figure()
ax = fig.gca()
ax.imshow(JNPIMG2[0,:,:,100,0])
ax.axis('off')
ax.set_title('3D conv input');

fig = plt.figure()
ax = fig.gca()
ax.imshow(out[0,:,:,100,0])
ax.axis('off')
ax.set_title('3D conv output');
plt.show()
