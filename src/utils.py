import torch
import config


# Create mapping dictionaries based on the character set in config
char_to_index = {char: index for index, char in enumerate(config.CHARACTER_SET)}
index_to_char = {index: char for index, char in enumerate(config.CHARACTER_SET)}

# --- Encoding/Decoding Functions ---

def encode_label(label_string: str) -> torch.Tensor:
    """
    Encodes a captcha string label into a tensor of numerical indices.

    Args:
        label_string: The string label (e.g., "aB3dE5").

    Returns:
        A tensor of shape (CAPTCHA_LENGTH,) containing character indices.
        Returns None if the label length doesn't match config.CAPTCHA_LENGTH
        or if it contains characters not in the CHARACTER_SET.
    """
    if len(label_string) != config.CAPTCHA_LENGTH:
        print(f"Error: Label '{label_string}' has incorrect length {len(label_string)}, expected {config.CAPTCHA_LENGTH}.")
        return None

    try:
        # Convert each character to its index
        encoded = [char_to_index[char] for char in label_string]
        # Using torch.long for indices suitable for embedding layers or loss functions
        return torch.tensor(encoded, dtype=torch.long)
    except KeyError as e:
        print(f"Error: Character '{e}' in label '{label_string}' not found in CHARACTER_SET.")
        return None


def decode_prediction(prediction_tensor: torch.Tensor) -> str:
    """
    Decodes a tensor of predicted indices back into a captcha string.
    Assumes the input tensor contains indices after an argmax operation.

    Args:
        prediction_tensor: A tensor of shape (CAPTCHA_LENGTH,) containing predicted indices.

    Returns:
        The decoded string prediction.
    """
    # Ensure tensor is on CPU and converted to list of Python integers
    indices = prediction_tensor.cpu().numpy().tolist()
    # Convert each index back to its character, EXCEPT: ? for unknown
    decoded_chars = [index_to_char.get(index, '?') for index in indices]
    return "".join(decoded_chars)

def decode_output_batch(output_batch: torch.Tensor) -> list[str]:
    """
    Decodes a batch of model outputs (after softmax and argmax) into strings.

    Args:
        output_batch: A tensor of shape (batch_size, CAPTCHA_LENGTH)
                      containing predicted indices for the batch.

    Returns:
        A list of decoded string predictions for the batch.
    """
    return [decode_prediction(prediction) for prediction in output_batch]


# --- test ---
