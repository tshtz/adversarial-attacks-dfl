import numpy as np
from nptyping import Float, NDArray, Shape


def reshape_arrays_for_adv_error_calc(
    *arrays: NDArray[Shape["* batchsize, ..."], Float],
) -> tuple[NDArray[Shape["* batchsize, * total_pred_param_nr"], Float], ...]:
    """A utility function to reshape the input arrays for adversarial error calculations."""
    # First assert that all the inputs have the same sizes
    for arr in arrays:
        assert arr.shape[0] == arrays[0].shape[0], "All arrays must have the same batch size"
    # Next reshape all the arrays
    reshaped_arrays = tuple(arr.reshape(arr.shape[0], -1) for arr in arrays)
    return reshaped_arrays


def get_f_value(
    dec: NDArray[Shape["* batchsize, ..."], Float], c: NDArray[Shape["* batchsize, ..."], Float]
) -> NDArray[Shape["* batchsize"], Float]:
    """Returns the f value for the given decision and input parameters.

    :param dec: The (batched) decision values
    :type dec: NDArray[Shape["* batchsize, ..."], Float]
    :param c: The (batched) parameters
    :type c: NDArray[Shape["* bauchsize, ..."], Float]
    :return: The f value for each sample in the batch
    :rtype: NDArray[Shape["* batchsize"], Float]
    """
    if dec.ndim > 2:
        # Reshape the arrays to ensure they are 2D
        dec, c = reshape_arrays_for_adv_error_calc(dec, c)
    return (dec * c).sum(axis=1)


def adv_accuracy_error(
    c_hat_adv: NDArray[Shape["* batchsize, ..."], Float],
    c: NDArray[Shape["* batchsize, ..."], Float],
    q: int,
) -> NDArray[Shape["* batchsize"], Float]:
    """Returns the accuracy error of the adversarial examples for each of the samples in the batch

    :param c_hat_adv: The (batched) predicted values for the adversarial examples
    :type c_hat_adv: NDArray[Shape["* batchsize, ..."], Float]
    :param c: The (batched) real values for the original examples
    :type c: NDArray[Shape["* batchsize, ..."], Float]
    :param q: The order of the norm
    :type q: int
    :return: The accuracy error for each sample in the batch
    :rtype: NDArray[Shape["* batchsize"], Float]
    """
    # Assert that the shapes of the inputs are correct
    if c_hat_adv.ndim > 2:
        # Reshape the arrays to ensure they are 2D
        c_hat_adv, c = reshape_arrays_for_adv_error_calc(c_hat_adv, c)
    return np.linalg.norm(c_hat_adv - c, ord=q, axis=1)


def adv_fooling_error(
    c_hat_adv: NDArray[Shape["* batchsize,  ..."], Float],
    c_hat: NDArray[Shape["* batchsize,  ..."], Float],
    q: int,
) -> NDArray[Shape["* batchsize"], Float]:
    """Returns the fooling error of the adversarial examples for each of the samples in the batch.

    :param c_hat_adv: The (batched) predicted values for the adversarial examples
    :type c_hat_adv: NDArray[Shape["* batchsize,  ..."], Float]
    :param c: The (batched) predicted values for the original examples
    :type c: NDArray[Shape["* batchsize,  ..."], Float]
    :param q: The order of the norm
    :type q: int
    :return: The fooling error for each sample in the batch
    :rtype: NDArray[Shape["* batchsize"], Float]
    """
    if c_hat_adv.ndim > 2:
        # Reshape the arrays to ensure they are 2D
        c_hat_adv, c_hat = reshape_arrays_for_adv_error_calc(c_hat_adv, c_hat)
    return np.linalg.norm(c_hat_adv - c_hat, ord=q, axis=1)


def adv_absolute_regret_error(
    c: NDArray[Shape["* batchsize, ..."], Float],
    dec_adv: NDArray[Shape["* batchsize, ..."], Float],
    dec_adv_hat: NDArray[Shape["* batchsize, ..."], Float],
    minimize: bool = True,
) -> NDArray[Shape["* batchsize"], Float]:
    """Returns the relative regret error of the adversarial examples for each of the samples in the
    batch.

    :param c: The (batched) real input parameters
    :type c: NDArray[Shape["* batchsize, ..."], Float]
    :param dec_adv: The (batched) real/optimal decision values for the adversarial examples
        (same as for the original examples)
    :type dec_adv: NDArray[Shape["* batchsize, ..."], Float]
    :param dec_adv_hat: The (batched) predicted decision values for the adversarial examples
    :type dec_adv_hat: NDArray[Shape["* batchsize, ..."], Float]
    :param minimize: Whether the problem is a minimization problem or not
    :type minimize: bool
    :return: The relative regret error for each sample in the batch
    :rtype: NDArray[Shape["* batchsize"], Float]
    """
    # Assert that the shapes of the inputs are correct
    if c.ndim > 2:
        # Reshape the arrays to ensure they are 2D
        c, dec_adv, dec_adv_hat = reshape_arrays_for_adv_error_calc(c, dec_adv, dec_adv_hat)
    mm = 1 if minimize else -1
    return mm * (((dec_adv_hat * c).sum(1)) - ((dec_adv * c).sum(1)))


def adv_relative_regret_error(
    c: NDArray[Shape["* batchsize, ..."], Float],
    dec_adv: NDArray[Shape["* batchsize, ..."], Float],
    dec_adv_hat: NDArray[Shape["* batchsize, ..."], Float],
    minimize: bool = True,
) -> NDArray[Shape["* batchsize"], Float]:
    """Returns the relative regret error of the adversarial examples for each of the samples in the
    batch.

    :param c: The (batched) real input parameters
    :type c: NDArray[Shape["* batchsize, ..."], Float]
    :param dec_adv: The (batched) real/optimal decision values for the adversarial examples
        (same as for the original examples)
    :type dec_adv: NDArray[Shape["* batchsize, ..."], Float]
    :param dec_adv_hat: The (batched) predicted decision values for the adversarial examples
    :type dec_adv_hat: NDArray[Shape["* batchsize, ..."], Float]
    :param minimize: Whether the problem is a minimization problem or not
    :type minimize: bool
    :return: The relative regret error for each sample in the batch
    :rtype: NDArray[Shape["* batchsize"], Float]
    """
    if c.ndim > 2:
        # Reshape the arrays to ensure they are 2D
        c, dec_adv, dec_adv_hat = reshape_arrays_for_adv_error_calc(c, dec_adv, dec_adv_hat)
    mm = 1 if minimize else -1
    return (mm * ((dec_adv_hat * c).sum(1) - ((dec_adv * c).sum(1)))) / (dec_adv * c).sum(1)


def adv_fooling_relative_regret_error(
    c: NDArray[Shape["* batchsize, ..."], Float],
    dec_adv: NDArray[Shape["* batchsize, ..."], Float],
    dec_hat: NDArray[Shape["* batchsize, ..."], Float],
    dec_adv_hat: NDArray[Shape["* batchsize, ..."], Float],
    minimize: bool = True,
):
    """Returns the fooling relative regret error of the adversarial examples for each of the samples
    in the batch.

    :param c: The (batched) real input parameters
    :type c: NDArray[Shape["* batchsize, ..."], Float]
    :param dec_adv: The (batched) real/optimal decision values for the adversarial examples
        (same as for the original examples)
    :type dec_adv: NDArray[Shape["* batchsize, ..."], Float]
    :param dec_hat: The (batched) predicted decision values for the original examples
    :type dec_hat: NDArray[Shape["* batchsize, ..."], Float]
    :param dec_adv_hat: The (batched) predicted decision values for the adversarial examples
    :type dec_adv_hat: NDArray[Shape["* batchsize, ..."], Float]
    :param minimize: Whether the problem is a minimization problem or not
    :type minimize: bool
    :return: The relative regret error for each sample in the batch
    :rtype: NDArray[Shape["* batchsize"], Float]
    """
    rel_regret_error = adv_relative_regret_error(
        c=c, dec_adv=dec_adv, dec_adv_hat=dec_adv_hat, minimize=minimize
    )
    # use the same formula but now with the predicted dec_hat
    rel_regret_error_non_adv = adv_relative_regret_error(
        c=c, dec_adv=dec_adv, dec_adv_hat=dec_hat, minimize=minimize
    )
    # Now compute the difference
    diffs = rel_regret_error - rel_regret_error_non_adv
    return diffs
