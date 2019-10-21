; nasm assmebly rountines for the core of the scancount methods

BITS 64

; must be equal to the unroll value in fastscancount_avx2b.h
%define UNROLL 16

; keep in sync with aux_chunk in fastscancount_avx2b.h
struc aux_chunk
        .start_ptr:  resb 8
        .iter_count: resb 4
        .overshoot:  resb 4 ; not used
endstruc

; the counters global array
extern _ZN13fastscancount8countersE

%define COUNTER_ARRAY abs _ZN13fastscancount8countersE

global record_hits_asm_branchy:function,record_hits_asm_branchless:function


; rdi : const uint32_t** aux_ptr
; rsi : const uint32_t** aux_end
; rdx : uint32_t start
; rcx : const implb::all_data &data (unused)
record_hits_asm_branchy:
        mov     rax, [rdi + aux_chunk.start_ptr] ; load eptr
        mov     r9d, [rdi + aux_chunk.iter_count]  ; load loop count
        mov     r10, [rdi + aux_chunk_size + aux_chunk.start_ptr]   ; load next eptr for prefetching

        jmp .top
ALIGN   32
.top:

%assign i 0
%rep UNROLL
        mov     r8d, [rax + i * 4]
        sub     r8d, edx
        add     byte [COUNTER_ARRAY + r8], 1
%if     i == 0
        prefetcht0 [rax + 128]
        prefetcht2 [r10]
%endif
%assign i (i + 1)
%endrep

        add     rax, 4 * UNROLL
        add     r10, 4 * UNROLL

        dec     r9d
        jnz     .top

; next array
        add     rdi, aux_chunk_size ; aux_ptr++
        cmp     rsi, rdi            ; break if aux_ptr == aux_end
        je      .done
        mov     rax, [rdi + aux_chunk.start_ptr] ; load eptr
        mov     r9d, [rdi + aux_chunk.iter_count]  ; load loop count

        ; prefetch the first few lines (also kicks off the L2 prefetcher)
%assign offset 0
%rep 2
        prefetchnta [rax + offset]
%assign offset (offset + 64)
%endrep

        ; touch the last page
        ;imul r10, r9, 4 * UNROLL
        ;prefetcht0 [rax + r10]

        ; prefetch the *next* array after this one
        mov     r10, [rdi + aux_chunk_size + aux_chunk.start_ptr]

        jmp     .top
.done:
        ret

; rdi : const uint32_t** aux_ptr
; rsi : const uint32_t** aux_end
; rdx : uint32_t start
; rcx : const implb::all_data &data (unused)
record_hits_asm_branchless:

        mov     rax, [rdi + aux_chunk.start_ptr]  ; load eptr
        mov     r9d, [rdi + aux_chunk.iter_count] ; load loop count
        mov     r10, [rdi + aux_chunk_size + aux_chunk.start_ptr]

ALIGN   32
.top:
%assign i 0
%rep UNROLL
        mov     r8d, dword [rax + i * 4]
        sub     r8d, edx
        add     byte [COUNTER_ARRAY + r8], 1
%assign i (i + 1)
%endrep

        add     rax, 4 * UNROLL

        lea     r8,  [rdi + aux_chunk_size]
        dec     r9

        cmovz   rax, [rdi + aux_chunk_size + aux_chunk.start_ptr]      ; eptr = next ? *(start_ptr + 1) : eptr
        cmovz   rdi, r8                                                ; if next start_ptr++
        cmovz   r9d, [rdi + aux_chunk.iter_count]                      ; iters = *loop_count
        mov     r10, [rdi + aux_chunk_size + aux_chunk.start_ptr]      ; next start_ptr (for prefetch)

%assign offset 0
%rep 2
        prefetchnta [rax + offset]
%assign offset (offset + 64)
%endrep

%assign offset 0
%rep 7
        prefetcht0 [r10 + offset]
%assign offset (offset + 64)
%endrep


; next array
        cmp     rsi, rdi          ; break if aux_ptr == aux_end
        jne      .top             ;

        ret


