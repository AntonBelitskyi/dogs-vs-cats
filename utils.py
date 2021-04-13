import magic


WHITE_MIME_TYPE_LIST = [
    'image/jpeg',
    'image/png'
]


class WrongMimeTypeError(Exception):
    pass


def check_mimetype(file):
    """
    Checking file extension
    :param file: str or PIL.Image
    :return: None
    """
    if isinstance(file, str):
        m_type = magic.from_file(file, mime=True)
    else:
        m_type = magic.from_buffer(file.read(4096), mime=True)
    if m_type not in WHITE_MIME_TYPE_LIST:
        raise WrongMimeTypeError
    return m_type
