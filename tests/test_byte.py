from assocmem.byte import ByteEncoderConfig, SignedHashByteEncoder, byte_split


def test_byte_encoder_and_split():
    encoder = SignedHashByteEncoder(
        ByteEncoderConfig(dimension=128, context_length=8, max_features=6)
    )
    query = encoder.encode(b"abcdefghijk")
    assert query.nnz <= 6
    assert byte_split(100) == {
        "train": (0, 80),
        "validation": (80, 90),
        "test": (90, 100),
    }
