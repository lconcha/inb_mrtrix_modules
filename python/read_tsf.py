import struct
import numpy as np

filename = "/misc/sherrington/lconcha/TMP/glaucoma/fs_glaucoma/sub-79864/dwi/rh_fsLR-32k_fa.tsf"


def read_tsf(filename):
    with open(filename, 'rb') as f:
        # Read the header
        header = {}
        while True:
            line = f.readline().decode('utf-8').strip()
            if line == "END":
                break
            if ':' not in line:
                continue  # skip lines without key:value
            key, value = line.split(':')
            header[key.strip()] = value.strip()

        # Extract metadata
        datatype = header['datatype']
        if datatype == 'Float32LE':
            dtype = np.dtype('<f4')
        elif datatype == 'Float64LE':
            dtype = np.dtype('<f8')
        else:
            raise ValueError(f"Unsupported datatype: {datatype}")
        

        # Get the offset from the 'file' key
        file_key = header.get('file')
        if file_key is None:
            raise ValueError("No 'file' key found in header.")
        offset = int(file_key.split()[1])

        # Seek to the start of the binary data
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype)

    # Parse streamlines using NaN and Inf
    streamlines = []
    current = []

    for val in data:
        if np.isnan(val):
            if current:
                streamlines.append(np.array(current))
                current = []
        elif np.isinf(val):
            if current:
                streamlines.append(np.array(current))
            break
        else:
            current.append(val)

    return header, streamlines

