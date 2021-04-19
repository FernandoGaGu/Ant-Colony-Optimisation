# Module containing the elements necessary to serialise objects.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import pickle
import os
import traceback
from datetime import datetime


def serialize(obj: object, path: str, extension: str = None):
    """
    Method to serialize any type of object received as an argument.

    Parameters
    ----------
    obj: object
        Object to be serialized.

    path: str
        Path where the object will be saved.

    extension: str (default None)
        Extension to be added.
    """
    abs_path = os.path.abspath(path)

    while os.path.exists(abs_path):  # If exists update the name
        path_to_file = '/'.join(abs_path.split('/')[:-1])
        file_name = abs_path.split('/')[-1].split('.')[0]
        abs_path = '%s/%s_(%s)' % \
                   (path_to_file, file_name, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

        print('\nFile: %s already exists, name changed to %s' % (path, abs_path), end='')
        if extension is not None:
            print('.%s\n' % extension)
        else:
            print()

    # Add extension
    if extension is not None:
        if not abs_path.endswith('.%s' % extension):
            abs_path += '.%s' % extension

    with open(abs_path, 'ab') as out:
        pickle.dump(obj, out)

    print('%s object serialised correctly' % type(obj))


def load(path: str) -> object:
    """
    Method to deserialize the object stored in the file received as argument.

    Parameters
    ----------
    :param path: str
        Path where the object was saved.
    """
    abs_path = os.path.abspath(path)
    assert os.path.exists(abs_path), 'File %s not found.' % abs_path
    try:
        with open(path, 'rb') as input_file:
            obj = pickle.load(input_file)

        return obj
    except pickle.UnpicklingError as e:
        print(traceback.format_exc(e))
        raise
    except (AttributeError, EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
        raise
    except Exception as e:
        print(traceback.format_exc(e))
        raise
