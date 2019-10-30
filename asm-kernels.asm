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

global record_hits_asm_branchy32:function,record_hits_asm_branchless32:function
global record_hits_asm_branchy16:function,record_hits_asm_branchless16:function


; %1 suffix
; %2 load instruction (eg mov or movzx)
; %3 load size (eg dword or word)
%macro make_branchy 3
%define DSIZE (%1 / 8)
; rdi : const uint32_t** aux_ptr
; rsi : const uint32_t** aux_end
; rdx : uint32_t start
record_hits_asm_branchy%1:
        mov     rax, [rdi + aux_chunk.start_ptr] ; load eptr
        mov     r9d, [rdi + aux_chunk.iter_count]  ; load loop count
        mov     r10, [rdi + aux_chunk_size + aux_chunk.start_ptr]   ; load next eptr for prefetching

        jmp .top
ALIGN   32
.top:

%assign i 0
%rep UNROLL
        %2     r8d, %3 [rax + i * DSIZE]
        add     byte [COUNTER_ARRAY + r8], 1
%if     i == 0
        prefetcht0 [rax + 256]
        prefetcht0 [r10]
%endif
%assign i (i + 1)
%endrep

        add     rax, DSIZE * UNROLL
        add     r10, DSIZE * UNROLL

        dec     r9d
        jnz     .top

; next array
        add     rdi, aux_chunk_size ; aux_ptr++
        cmp     rsi, rdi            ; break if aux_ptr == aux_end
        je      .done
        mov     rax, [rdi + aux_chunk.start_ptr]  ; load eptr
        mov     r9d, [rdi + aux_chunk.iter_count] ; load loop count

        ; prefetch the first few lines (also kicks off the L2 prefetcher)
%assign offset 0
%rep 2
        prefetcht0 [rax + offset]
%assign offset (offset + 64)
%endrep

        ; prefetch the *next* array after this one
        mov     r10, [rdi + aux_chunk_size + aux_chunk.start_ptr]

        jmp     .top
.done:
        ret

%endmacro

make_branchy 32, mov  , dword
make_branchy 16, movzx, word

global record_hits_asm_branchyB:function
; rdi : const uint32_t** aux_ptr
; rsi : const uint32_t** aux_end
; rdx : uint32_t start
record_hits_asm_branchyB:
        push    r12
        push    r13
        push    r14
        mov     rax, [rdi + aux_chunk.start_ptr] ; load eptr
        mov     r9d, [rdi + aux_chunk.iter_count]  ; load loop count
        mov     r10, [rdi + aux_chunk_size + aux_chunk.start_ptr]   ; load next eptr for prefetching
        mov     r12d, 16
        mov     r13d, 32
        mov     r14d, 48

        jmp .top
ALIGN   32
.top:

%assign i 0
%rep UNROLL
%if i % 4 == 0
        mov     r11, QWORD [rax + i * DSIZE]
        mov     rcx, r11
%elif i % 4 == 1
        shrx    rcx, r11, r12
%elif i % 4 == 2
        shrx    rcx, r11, r13
%elif i % 4 == 3
        shrx    rcx, r11, r14
%endif
        movzx   r8d, cx
        add     byte [COUNTER_ARRAY + r8], 1
%if     i == 0
        prefetcht0 [rax + 256]
        prefetcht0 [r10]
%endif
%assign i (i + 1)
%endrep

        add     rax, DSIZE * UNROLL
        add     r10, DSIZE * UNROLL

        dec     r9d
        jnz     .top

; next array
        add     rdi, aux_chunk_size ; aux_ptr++
        cmp     rsi, rdi            ; break if aux_ptr == aux_end
        je      .done
        mov     rax, [rdi + aux_chunk.start_ptr]  ; load eptr
        mov     r9d, [rdi + aux_chunk.iter_count] ; load loop count

        ; prefetch the first few lines (also kicks off the L2 prefetcher)
%assign offset 0
%rep 2
        prefetcht0 [rax + offset]
%assign offset (offset + 64)
%endrep

        ; prefetch the *next* array after this one
        mov     r10, [rdi + aux_chunk_size + aux_chunk.start_ptr]

        jmp     .top

.done:
        pop    r14
        pop    r13
        pop    r12
        ret


; %1 suffix
; %2 load instruction (eg mov or movzx)
; %3 load size (eg dword or word)
%macro make_branchless 3
%define DSIZE (%1 / 8)
; rdi : const uint32_t** aux_ptr
; rsi : const uint32_t** aux_end
; rdx : uint32_t start
record_hits_asm_branchless%1:

        mov     rax, [rdi + aux_chunk.start_ptr]  ; load eptr
        mov     r9d, [rdi + aux_chunk.iter_count] ; load loop count

ALIGN   32
.top:
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

%assign i 0
%rep UNROLL
        %2     r8d, %3 [rax + i * DSIZE]
        add     byte [COUNTER_ARRAY + r8], 1
%assign i (i + 1)
%endrep

        add     rax, DSIZE * UNROLL

        lea     r8,  [rdi + aux_chunk_size]
        dec     r9

        cmovz   rax, [rdi + aux_chunk_size + aux_chunk.start_ptr]      ; eptr = next ? *(start_ptr + 1) : eptr
        cmovz   rdi, r8                                                ; if next start_ptr++
        cmovz   r9d, [rdi + aux_chunk.iter_count]                      ; iters = *loop_count

; next array
        cmp     rsi, rdi          ; break if aux_ptr == aux_end
        jne      .top             ;

        ret
%endmacro


make_branchless 32, mov  , dword
make_branchless 16, movzx, word

