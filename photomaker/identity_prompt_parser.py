import torch


def extract_identity_prompt_map_clean(
    text_input_ids,
    identity_token_indices,
    strict=True,
    debug=False,
):
    """
    Build identity → prompt token routing map.

    Args:
        text_input_ids: list[int] or 1D tensor
        identity_token_indices: list[list[int]]
            Example:
                [
                    [5],     # img1 token position(s)
                    [12]     # img2 token position(s)
                ]

        strict (bool):
            If True → raises error when identity order is invalid.
            If False → silently skips invalid identities.

        debug (bool):
            If True → prints routing info.

    Returns:
        identity_prompt_map: dict[int, list[int]]
            { identity_index: [token_indices...] }
    """

    # -------------------------------------------------
    # 1️⃣ Normalize input
    # -------------------------------------------------

    if isinstance(text_input_ids, torch.Tensor):
        text_input_ids = text_input_ids.tolist()

    seq_len = len(text_input_ids)

    if seq_len == 0:
        return {}

    if identity_token_indices is None:
        return {}

    if len(identity_token_indices) == 0:
        return {}

    identity_prompt_map = {}

    # -------------------------------------------------
    # 2️⃣ Sort identities by first occurrence (safety)
    # -------------------------------------------------

    cleaned_indices = []

    for idx_list in identity_token_indices:
        if idx_list is None or len(idx_list) == 0:
            cleaned_indices.append([])
            continue

        sorted_positions = sorted(idx_list)
        cleaned_indices.append(sorted_positions)

    # Sort identities by first token position
    identity_order = sorted(
        range(len(cleaned_indices)),
        key=lambda i: cleaned_indices[i][0]
        if len(cleaned_indices[i]) > 0 else float("inf")
    )

    cleaned_indices = [cleaned_indices[i] for i in identity_order]

    # -------------------------------------------------
    # 3️⃣ Build routing map
    # -------------------------------------------------

    for i, token_positions in enumerate(cleaned_indices):

        if len(token_positions) == 0:
            continue

        # Start routing AFTER last identity token
        start = max(token_positions) + 1

        # End before next identity
        if i < len(cleaned_indices) - 1:
            next_positions = cleaned_indices[i + 1]

            if len(next_positions) == 0:
                end = seq_len
            else:
                end = min(next_positions)
        else:
            end = seq_len

        if start >= end:
            if strict:
                raise ValueError(
                    f"Invalid routing span for identity {i}: "
                    f"start={start}, end={end}"
                )
            else:
                continue

        identity_prompt_map[i] = list(range(start, end))

        if debug:
            print(
                f"Identity {i} routes tokens: "
                f"{start} → {end - 1}"
            )

    return identity_prompt_map
