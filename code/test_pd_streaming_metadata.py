import unittest

from pd_streaming_metadata import (
    PushChunk,
    chunk_ranges,
    has_write_capacity,
    is_partial_completion_notif,
    slice_block_ids,
)


class StreamingMetadataTests(unittest.TestCase):
    def test_slice_destination_blocks_uses_offset_for_each_group(self):
        registered = ([10, 11, 12, 13], [20, 21, 22, 23])
        self.assertEqual(slice_block_ids(registered, 2, (2, 1)), ([12, 13], [22]))


    def test_slice_destination_blocks_rejects_short_registration(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            slice_block_ids(([10, 11],), 1, (2,))


    def test_final_flag_and_completion_notification_filter(self):
        partial = PushChunk("req", 0, ([10, 11],), 3, False)
        final = PushChunk("req", 2, ([12],), 3, True)
        self.assertFalse(partial.is_final)
        self.assertTrue(final.is_final)
        self.assertTrue(is_partial_completion_notif(b"req#chunk-0:1"))
        self.assertFalse(is_partial_completion_notif(b"req:1"))

    def test_long_final_tail_is_contiguous_and_c8_bounded(self):
        ranges = chunk_ranges(128, 563, 8)
        self.assertEqual(ranges[0], (128, 136))
        self.assertEqual(ranges[-1], (560, 563))
        self.assertEqual(ranges[0][0], 128)
        for previous, current in zip(ranges, ranges[1:]):
            self.assertEqual(previous[1], current[0])
        self.assertTrue(all(end - start <= 8 for start, end in ranges))

    def test_write_capacity_enforces_configured_handle_limit(self):
        self.assertTrue(has_write_capacity(1, 0))
        self.assertFalse(has_write_capacity(1, 1))
        self.assertTrue(has_write_capacity(2, 1))
        self.assertTrue(has_write_capacity(0, 2))  # zero means unlimited


if __name__ == "__main__":
    unittest.main()
