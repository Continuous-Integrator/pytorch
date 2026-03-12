import torch


# Used for testing and logging
dtype_abbrs = {dt: dt.abbr for dt in [
    torch.bfloat16, torch.float64, torch.float32, torch.float16,
    torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2,
    torch.complex32, torch.complex64, torch.complex128,
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.bool, torch.uint8, torch.uint16, torch.uint32, torch.uint64,
    torch.bits16, torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8,
]}
