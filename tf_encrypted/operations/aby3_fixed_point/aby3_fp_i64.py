import logging
import os
import tensorflow as tf
import tf_encrypted as tfe

SO_PATH = "{dn}/operations/aby3_fixed_point/aby3_fp_module_tf_{tfv}.so"
logger = logging.getLogger('tf_encrypted')

def _try_load_aby3_fp_module():
    """
    Attempt to load and return aby3_fp module; returns None if failed.
    """
    so_file = SO_PATH.format(dn=os.path.dirname(tfe.__file__), tfv=tf.__version__)
    if not os.path.exists(so_file):
        logger.warning(
            (
                "Falling back to insecure randomness since the required custom op "
                "could not be found for the installed version of TensorFlow. Fix "
                "this by compiling custom ops. Missing file was '%s'"
            ),
            so_file,
        )
        return None

    try:
        return tf.load_op_library(so_file)

    except NotFoundError as ex:
        logger.warning(
            (
                "Falling back to insecure randomness since the required custom op "
                "could not be found for the installed version of TensorFlow. Fix "
                "this by compiling custom ops. "
                "Missing file was '%s', error was \"%s\"."
            ),
            so_file,
            ex,
        )

    except Exception as ex:  # pylint: disable=broad-except
        logger.error(
            (
                "Falling back to insecure randomness since an error occurred "
                'loading the required custom op: "%s".'
            ),
            ex,
        )

    return None

aby3_fp_module = _try_load_aby3_fp_module()

def _is_tf_tensor(x):
    return isinstance(x, (tf.Tensor, tf.Variable))

def i64_bit_reverse(x: tf.Tensor):
    """
    bit-reverse the uint64 tensor
    """
    assert _is_tf_tensor(x)
    return aby3_fp_module.i64_bit_reverse(x)

def i64_bit_gather(x: tf.Tensor, even=True):
    """
    Gather bits of even (or odd) positions.
    For example Gather((b0, b1, b2, b3)_2, even = True) => (b0, b2)_2
                Gather((b0, b1, b2, b3)_2, even = False) => (b1, b3)_2
    """
    assert _is_tf_tensor(x)
    return aby3_fp_module.i64_bit_gather(x, even)

def i64_xor_indices(x: tf.Tensor):
    """
    XOR the positions with non-zero.
    For bit sequence, b0,b1,b2,...,b_63, compute XOR_{i: b_i = 1}(i)
    """
    assert _is_tf_tensor(x)
    return aby3_fp_module.i64_xor_indices(x)

