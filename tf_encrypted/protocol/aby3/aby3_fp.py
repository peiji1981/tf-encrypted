# This file include the  floating-point operations
import tensorflow as tf
import tf_encrypted as tfe
from math import ceil, log2
import numpy as np

def prefix_ORs(b: "PrivateTensor", k: int):
    import math
    """
    b := (..., b3, b2, b1, b0) where b0 is the least significant bit
    compute y := (y_{k-1}, y_{k-1}, ..., y0) where y_i \in {0, 1}
         y_i = bit_or_{i <= j < k} bj
         The first yi = 1 is the first significant bit that bi = 1.

    """
    ## running ORs from MSB to LSB.
    with tf.name_scope("prefix_ORs"):
        n = int(ceil(log2(k))) + 1
        e = 1
        for i in range(1, n):
            b = b | (b >> e)
            e = e << 1
    return b

def bit_reverse(prot: "ABY3", x: "ABY3PrivateTensor"):
    shares = x.unwrapped
    results = [[None] * 2 for _ in range(3)]
    with tf.name_scope("bit_reverse"):
        for idx in range(3):
            with tf.device(prot.servers[idx].device_name):
                results[idx][0] = shares[idx][0].bit_reverse()
                results[idx][1] = shares[idx][1].bit_reverse()
    return tfe.protocol.aby3.ABY3PrivateTensor(prot, results, x.is_scaled, x.share_type)

def gather_bit(prot: "ABY3", x: "ABY3PrivateTensor", even=True):
    shares = x.unwrapped
    z = [[None, None], [None, None], [None, None]]
    for idx in range(3):
        with tf.device(prot.servers[idx].device_name):
            z[idx][0] = shares[idx][0].bit_gather(even)
            z[idx][1] = shares[idx][1].bit_gather(even)
    return tfe.protocol.aby3.ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)

def xor_indices(prot: "ABY3", x: "ABY3PrivateTensor"):
    shares = x.unwrapped
    z = [[None, None], [None, None], [None, None]]
    for idx in range(3):
        with tf.device(prot.servers[idx].device_name):
            z[idx][0] = shares[idx][0].xor_indices()
            z[idx][1] = shares[idx][1].xor_indices()
    return tfe.protocol.aby3.ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)

def _fp_sqrt_private(prot: "ABY3", a: "ABY3PrivateTensor"):
    n_iterations = 3
    c15 = prot.define_constant(1.5)
    c05 = prot.define_constant(0.5)

    y = approx_sqrt_inv(prot, a, False) # y approixates 1 / sqrt(a)
    g = a * y
    h = y * c05
    """
    g -> sqrt(b)
    h -> sqrt(1 / b) * 0.5
    """
    for _ in range(n_iterations):
        r = c15 - g * h
        g = g * r
        h = h * r
    return g

def _fp_sqrt_inv_private(prot: "ABY3", a: "ABY3PrivateTensor"):
    return approx_sqrt_inv(prot, a, True)

"""
    From x, to compute an approximation of 1/sqrt(x).
    Let x = b * 2^j for some value 0.25 <= b  < 0.5 and some integer j.
    1./ sqrt(x) can be approximated as f(b) * 2^{-j/2} using low-degree polynomail function f(b),
    e.g., f(b) = 4.6388720536918487 * b^2 -5.7778925299580213 * b + 3.1473614921017938 approixmates
          1/sqrt(b) with less than 0.35% relative error for b in [0.25, 0.5).
"""
def approx_sqrt_inv(prot, x: "PrivateTensor", high_precision=True):
    def select(x, y, bit):
        """
        return x if bit = 0 else y.
        """
        c = np.ones(shape = bit.shape) * (x - y)
        return prot.mul_AB(prot.define_constant(c, apply_scaling=False), bit) + y

    k = prot.fixedpoint_config.precision_fractional
    f = prot.fixedpoint_config.precision_integral
    m = k + f
    n = prot.int_factory.nbits
    assert x.is_scaled, "Require a fixed-point value"
    # We assume 2^{-j} can be represent with 2k-bit precisions for all j in [0, m)
    assert 2*k > m, "Use a larger precision_fractional"
    with tf.name_scope("normalization_for_sqrt"):
        x_bits = prot.A2B(x) # 0.5 * log2(n)
        y_bits = prefix_ORs(x_bits, n) # 0.5 * log2(n)
        z_bits = y_bits ^ (y_bits >> 1) ## note: x = c * 2^m where c \in [0.25, 0.5)

        rev_z_bits = bit_reverse(prot, z_bits) >> (n - 2*k)
        rev_z_bits_one_more = rev_z_bits >> 1

        frac = prot.B2A(rev_z_bits_one_more, 2*k)  # extra 1 for [0.5, 1) -> [0.25, 0.5) #0.5 * log2(2k)
        frac.is_scaled = True
        normalized = frac * x # normalized \in [0.25, 0.5) # 1
        """
        f(b) = 4.6388720536918487 * b^2 -5.7778925299580213 * b + 3.1473614921017938 approixmates 1/sqrt(b) in [0.25, 0.5)
        with less than 0.35% relative error
        """
        if high_precision: #  2 rounds
            sqrt_inv = ((4.6388720536918487 * normalized) -5.7778925299580213) * normalized + 3.1473614921017938
        else:
            sqrt_inv = -2.2563978299520072 * normalized + 2.5146075725942229
        """
            Indeed, the exponetent part is 2^{j+k} where k is the scaling factor.
            We want to compute sqrt(2^{-j}) with k-bit precision, i.e., sqrt(2^{-j}) * 2^k.
            In other words, we compute sqrt(2^{-j}) * 2^k from 2^{j+k}.

            1. We first obtain 2^{-(j+k)} from 2^{j + k}.
            2. Then we compute 2^{floor(-(j+k)/2)}. Rewrite it as
               2^{floor(-(j+k)/2)} = c * 2^{floor(-j/2)} * 2^{floor(-k/2)}
               where c depends on the parity of j, and k.
            3. We compute the parity of j + k, i.e., check the LSB of j + k.
            4. Suppose k is even, 2^{floor(-k/2)} = 2^{-k/2}. Then we can cancel this term via 2^{k/2}.
               If lsb(j + k) = 0 <-> j is even. In this case,
               2^{floor(-j/2)} = 2^{-j/2} = sqrt(2^{-j}).
               If lsb(j + k) = 1 <-> j is odd. Then
               2^{floor(-j/2)} * 2^{-1} = sqrt(2^{-j}).

               Suppose k is odd: We need 2^{k//2} * 2 to cancel 2^{floor(-k/2)}.
               If lsb(j + k) = 0 <-> j is odd. In this case,
               2^{floor(-j/2)} * 2^{-1} = sqrt(2^{-j}).
               If lsb(j + k) = 1 <-> j is odd. Then
               2^{floor(-j/2)} = 2^{-j/2} = sqrt(2^{-j}).
        """
        j_add_k = xor_indices(prot, z_bits) # j + k
        lsb = prot.bit_extract(j_add_k, 0) # lsb = 0 <-> j + k is even

        # 0.5 * log2(k)
        exponet = prot.B2A(gather_bit(prot, rev_z_bits | rev_z_bits_one_more, True), k) # exponet = 2^{floor((k - j)/2)}
        exponet.is_scaled = False # Stop truncation

        if k & 1 == 0: # k is even which means lsb = 1 <=> j is odd
            exponet = exponet * select(2**(k//2) / np.sqrt(2.), 2**(k//2) , lsb)
        else: # k is odd which means lsb = 1 <=> j is even
            exponet = exponet * select(2**(k//2), 2**(k//2 + 1) / np.sqrt(2.), lsb)
        exponet.is_scaled = True # 2^{-j/2} with k-bit precision

    return sqrt_inv * exponet

