import torch
import triton
import triton.language as tl

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 256
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    PID = tl.program_id(axis=0)
    #if vec is len 256, and block_size is 64, 
    # then PID 0 processes [0:64] 
    # PID 1 processes [64:128]
    block_start = PID* BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=None) #shape {BLOCK_SIZE}
    y = tl.load(y_ptr + offsets, mask=mask, other=None)

    output = x + y
    
    #write back to dram
    tl.store(output_ptr + offsets, output, mask=mask)
    
def add(x, y):
    assert x.device == y.device, "must be on the same device"
    assert x.shape == y.shape, "must be the same"
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),1)
    # grid = (int((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE), 1)
    print(f"grid: {grid}")
    
    add_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def test_add_kernel(size, abstol=1e-3, reltol=1e-3, device=DEVICE):
    torch.manual_seed(42)
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    
    z_ref = x+y 
    z_tri = add(x, y)
    
    torch.testing.assert_close(z_tri, z_ref, atol=abstol, rtol=reltol)
    print("passed") 
    
if __name__ == '__main__':

    test_add_kernel(size=1030)
    test_add_kernel(size=1024)
