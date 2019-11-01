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

GLOBAL override_middle_asm_3
override_middle_asm_3:
        push    rbp
        mov     rax, r8
        mov     rbp, rsp
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx

        mov     qword [rsp-8H], r8

%macro load_and_pf 2
        mov             %1, %2
        ;prefetcht0      [%1]
        ;prefetcht0      [%1 + 64]
%endmacro

        ; load 8 element pointers
        load_and_pf     r14,  [rdx]
        load_and_pf     r13,  [rdx+8H]
        load_and_pf     r12,  [rdx+10H]
        load_and_pf     rbx,  [rdx+18H]
        load_and_pf     r11,  [rdx+20H]
        load_and_pf     r10,  [rdx+28H]
        load_and_pf     r9,   [rdx+30H]
        load_and_pf     r8,   [rdx+38H]

        cmp     rcx, rax
        jnc     .done

        mov     rax, qword [rsi]
        mov     rax, qword [rax]
        mov     qword [rsp-38H], rax
        mov     rax, qword [rsi+8H]
        mov     r15, qword [rax]
        mov     rax, qword [rsi+10H]
        mov     qword [rsp-30H], r15
        mov     r15, qword [rax]
        mov     rax, qword [rsi+18H]
        mov     qword [rsp-28H], r15
        mov     r15, qword [rax]
        mov     rax, qword [rsi+20H]
        mov     qword [rsp-20H], r15
        mov     r15, qword [rax]
        mov     rax, qword [rsi+28H]
        mov     qword [rsp-18H], r15
        mov     r15, qword [rax]
        mov     rax, qword [rsi+30H]
        mov     qword [rsp-10H], r15
        mov     r15, qword [rax]
        mov     rax, qword [rsi+38H]
        mov     rsi, qword [rax]
        mov     rax, qword [rdi]
        mov     rdi, qword [rsp-38H]
        mov     qword [rsp-38H], rdx

ALIGN 64
.top:
        movzx   edx, word [rdi+rcx*2]
        add     rax, 256
        kmovw   k1, edx
        vpexpandd zmm2 {k1}{z}, zword [r14]
        popcnt  rdx, rdx
        lea     r14, [r14+rdx*4]

        mov     rdx, qword [rsp-30H]
        vmovdqa64 zmm0, zmm2
        movzx   edx, word [rdx+rcx*2]
        kmovw   k2, edx
        vpexpandd zmm5 {k2}{z}, zword [r13]
        popcnt  rdx, rdx
        lea     r13, [r13+rdx*4]

        mov     rdx, qword [rsp-28H]
        movzx   edx, word [rdx+rcx*2]
        kmovw   k3, edx
        vpexpandd zmm8 {k3}{z}, zword [r12]
        popcnt  rdx, rdx
        lea     r12, [r12+rdx*4]

        mov     rdx, qword [rsp-20H]
        movzx   edx, word [rdx+rcx*2]
        kmovw   k4, edx
        vpexpandd zmm3 {k4}{z}, zword [rbx]
        popcnt  rdx, rdx
        lea     rbx, [rbx+rdx*4]

        mov     rdx, qword [rsp-18H]
        movzx   edx, word [rdx+rcx*2]
        kmovw   k5, edx
        vpexpandd zmm6 {k5}{z}, zword [r11]
        popcnt  rdx, rdx
        lea     r11, [r11+rdx*4]

        mov     rdx, qword [rsp-10H]
        movzx   edx, word [rdx+rcx*2]
        kmovw   k6, edx
        vpexpandd zmm7 {k6}{z}, zword [r10]
        popcnt  rdx, rdx
        lea     r10, [r10+rdx*4]

        movzx   edx, word [r15+rcx*2]
        kmovw   k7, edx
        vpexpandd zmm1 {k7}{z}, zword [r9]
        popcnt  rdx, rdx
        lea     r9, [r9+rdx*4]

        movzx   edx, word [rsi+rcx*2]
        kmovw   k1, edx
        vpexpandd zmm4 {k1}{z}, zword [r8]
        popcnt  rdx, rdx
        lea     r8, [r8+rdx*4]

%ifndef pf_offset
%define pf_offset 256
%endif
%ifndef pf_instr
%define pf_instr prefetcht0
%endif

%warning 'pf offset' pf_offset ' pf_instr' pf_instr

        pf_instr [r14 + pf_offset]
        pf_instr [r13 + pf_offset]
        pf_instr [r12 + pf_offset]
        pf_instr [rbx + pf_offset]
        pf_instr [r11 + pf_offset]
        pf_instr [r10 + pf_offset]
        pf_instr [r9  + pf_offset]
        pf_instr [r8  + pf_offset]

        vpternlogd zmm2, zmm5, zmm8, 0x96
        vpternlogd zmm0, zmm5, zmm8, 0xE8

        vmovdqa64 zmm5, zmm3
        vpternlogd zmm5, zmm6, zmm7, 0xE8
        vpternlogd zmm3, zmm6, zmm7, 0x96

        vmovdqa64 zmm6, zmm1
        vpternlogd zmm1, zmm2, zmm3, 0x96
        vpternlogd zmm6, zmm2, zmm3, 0xE8

        vmovdqa64 zmm3, zmm0
        vpternlogd zmm3, zmm5, zmm6, 0xE8
        vpternlogd zmm0, zmm5, zmm6, 0x96

        vmovdqa64 zmm5, zword [rax-4H*40H]
        vmovdqa64 zmm2, zmm4
        vpternlogd zmm4, zmm1, zmm5, 0x96
        vpternlogd zmm2, zmm1, zmm5, 0xE8
        vmovdqa64 zword [rax-4H*40H], zmm4

        vmovdqa64 zmm4, zword [rax-3H*40H]
        vmovdqa64 zmm1, zmm2
        vpternlogd zmm2, zmm0, zmm4, 0x96
        vpternlogd zmm1, zmm0, zmm4, 0xE8
        vmovdqa64 zword [rax-3H*40H], zmm2

        vmovdqa64 zmm2, zword [rax-2H*40H]
        vmovdqa64 zmm0, zmm1
        vpternlogd zmm0, zmm3, zmm2, 0xE8
        vpord   zmm0, zmm0, zword [rax-1H*40H]
        vpternlogd zmm1, zmm3, zmm2, 0x96
        vmovdqa64 zword [rax-2H*40H], zmm1
        vmovdqa64 zword [rax-1H*40H], zmm0

        inc     rcx
        cmp     qword [rsp-8H], rcx
        jne     .top

        mov     rdx, qword [rsp-38H]

.done:
        mov     qword [rdx], r14
        mov     qword [rdx+8H], r13
        mov     qword [rdx+10H], r12
        mov     qword [rdx+18H], rbx
        mov     qword [rdx+20H], r11
        mov     qword [rdx+28H], r10
        mov     qword [rdx+30H], r9
        mov     qword [rdx+38H], r8

        lea     rsp, [rbp-28H]
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbp
        vzeroupper
        ret
