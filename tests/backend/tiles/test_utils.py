import numpy as np
import pytest

from invokeai.backend.tiles.utils import TBLR, paste, seam_blend


def test_paste_no_mask_success():
    """Test successful paste with mask=None."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)

    # Create src_image with a pattern that can be used to validate that it was pasted correctly.
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)
    src_image[0, :, 0] = 1  # Row of 1s in channel 0.
    src_image[:, 0, 1] = 2  # Column of 2s in channel 1.

    # Paste in bottom-center of dst_image.
    box = TBLR(top=2, bottom=5, left=1, right=4)

    # Construct expected output image.
    expected_output = np.zeros((5, 5, 3), dtype=np.uint8)
    expected_output[2, 1:4, 0] = 1
    expected_output[2:5, 1, 1] = 2

    paste(dst_image=dst_image, src_image=src_image, box=box)

    np.testing.assert_array_equal(dst_image, expected_output, strict=True)


def test_paste_with_mask_success():
    """Test successful paste with a mask."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)

    # Create src_image with a pattern that can be used to validate that it was pasted correctly.
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)
    src_image[0, :, 0] = 64  # Row of 64s in channel 0.
    src_image[:, 0, 1] = 128  # Column of 128s in channel 1.

    # Paste in bottom-center of dst_image.
    box = TBLR(top=2, bottom=5, left=1, right=4)

    # Create a mask that blends the top-left corner of 'src_image' at 50%, and ignores the rest of src_image.
    mask = np.zeros((3, 3))
    mask[0, 0] = 0.5

    # Construct expected output image.
    expected_output = np.zeros((5, 5, 3), dtype=np.uint8)
    expected_output[2, 1, 0] = 32
    expected_output[2, 1, 1] = 64

    paste(dst_image=dst_image, src_image=src_image, box=box, mask=mask)

    np.testing.assert_array_equal(dst_image, expected_output, strict=True)


@pytest.mark.parametrize("use_mask", [True, False])
def test_paste_box_overflows_dst_image(use_mask: bool):
    """Test that an exception is raised if 'box' overflows the 'dst_image'."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = None
    if use_mask:
        mask = np.zeros((3, 3))

    # Construct box that overflows bottom of dst_image.
    top = 3
    left = 0
    box = TBLR(top=top, bottom=top + src_image.shape[0], left=left, right=left + src_image.shape[1])

    with pytest.raises(ValueError):
        paste(dst_image=dst_image, src_image=src_image, box=box, mask=mask)


@pytest.mark.parametrize("use_mask", [True, False])
def test_paste_src_image_does_not_match_box(use_mask: bool):
    """Test that an exception is raised if the 'src_image' shape does not match the 'box' dimensions."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = None
    if use_mask:
        mask = np.zeros((3, 3))

    # Construct box that is smaller than src_image.
    box = TBLR(top=0, bottom=src_image.shape[0] - 1, left=0, right=src_image.shape[1])

    with pytest.raises(ValueError):
        paste(dst_image=dst_image, src_image=src_image, box=box, mask=mask)


def test_paste_mask_does_not_match_src_image():
    """Test that an exception is raised if the 'mask' shape is different than the 'src_image' shape."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)

    # Construct mask that is smaller than the src_image.
    mask = np.zeros((src_image.shape[0] - 1, src_image.shape[1]))

    # Construct box that matches src_image shape.
    box = TBLR(top=0, bottom=src_image.shape[0], left=0, right=src_image.shape[1])

    with pytest.raises(ValueError):
        paste(dst_image=dst_image, src_image=src_image, box=box, mask=mask)


@pytest.mark.parametrize("x_seam", [False, True])
def test_seam_blend_seam_follows_low_energy_band(x_seam: bool):
    """Test that seam_blend() puts the seam on the lowest-energy line.

    The seam's starting position is found with np.argmin() over a slice of the cumulative energy that
    starts at 'gutter' (ceil(blend_amount / 2)) rather than at index 0, so the result has to be
    offset back to an absolute index. Without that offset the seam is placed 'gutter' px short of the
    low-energy line it just found. x_seam rotates the search, so both orientations are covered.
    """
    seam_axis_size = 64  # The axis the seam is searched along.
    other_axis_size = 4
    blend_amount = 8
    band_center = 40  # Where the two images agree, i.e. the cheapest place to put the seam.

    # ia2 - ia1 is zero in a narrow band and a high-frequency checkerboard everywhere else, so the
    # lowest-energy seam is the band. A y-seam searches down columns, an x-seam across rows.
    shape = (seam_axis_size, other_axis_size) if x_seam else (other_axis_size, seam_axis_size)
    diff = np.zeros(shape, dtype=np.float64)
    for y in range(shape[0]):
        for x in range(shape[1]):
            if abs((y if x_seam else x) - band_center) > 1:
                diff[y, x] = 100.0 * ((x + y) % 2)

    ia1 = np.zeros((*shape, 3), dtype=np.float64)
    ia2 = np.repeat(diff[:, :, None], 3, axis=2)

    blended = seam_blend(ia1, ia2, blend_amount=blend_amount, x_seam=x_seam)

    # blended is ia1 * mask + ia2 * (1 - mask), and ia1 is zeros, so blended is diff * (1 - mask).
    # Averaging along the other axis cancels the checkerboard and recovers the mask, which is 1 on
    # one side of the seam and 0 on the other.
    unblended = diff.mean(axis=1 if x_seam else 0)
    textured = unblended > 0
    mask_profile = np.ones(seam_axis_size)
    mask_profile[textured] = 1.0 - blended.mean(axis=(1, 2) if x_seam else (0, 2))[textured] / unblended[textured]

    past_seam = np.flatnonzero(mask_profile < 0.5)
    assert len(past_seam) > 0, "the seam mask never transitions from ia1 to ia2"
    assert abs(int(past_seam[0]) - band_center) <= 2
