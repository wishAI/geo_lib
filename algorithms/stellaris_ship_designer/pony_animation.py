"""Read PonyLumen v0.4 binary model data without running downloaded JavaScript.

Layout verified against https://ponylumen.net/js/binaryModel_v.0.9.7.js.
Matrices use column-major order and include each part's ORIGINAL bind frame.
"""
import struct


class Reader:
    def __init__(self, data):
        self.data, self.position = data, 0

    def read(self, fmt, count=1):
        size = struct.calcsize('<' + fmt) * count
        if count < 0 or self.position + size > len(self.data):
            raise ValueError('Truncated or invalid PonyLumen model')
        values = struct.unpack_from('<' + fmt * count, self.data, self.position)
        self.position += size
        return values[0] if count == 1 else list(values)

    def array(self, fmt):
        count = self.read('i')
        result = self.read(fmt, count)
        return [result] if count == 1 else result

    def skip(self, count):
        if count < 0 or self.position + count > len(self.data):
            raise ValueError('Invalid data length')
        self.position += count

    def string(self):
        count = self.read('i')
        start = self.position
        self.skip(count)
        return self.data[start:self.position].decode('latin1')


def parse(path):
    r = Reader(path.read_bytes())
    version, original_count, vertex_count = r.read('f'), r.read('i'), r.read('i')
    if not any(abs(version - supported) < .001 for supported in (.2, .3, .4)):
        raise ValueError(f'Unsupported model version {version}')
    positions, extra, uv, counts = r.array('f'), r.array('i'), r.array('f'), r.array('B')
    indices, weights = (r.array('B'), r.array('f')) if counts else ([], [])
    if r.read('i'):
        for _ in range(original_count):
            r.skip(r.read('B') * 25)
    faces, joint_count = r.array('H'), r.read('i')
    result = dict(file=path.name, version=version, vertex_count=vertex_count, joint_count=joint_count)
    if joint_count:
        root = r.read('i')
        names = [r.string() for _ in range(r.read('i'))]
        parents, frames, matrices = r.array('i'), r.read('i'), r.array('f')
        if len(matrices) != joint_count * frames * 16:
            raise ValueError('Animation matrix count mismatch')
        result.update(root=root, names=names, parents=parents, frame_count=frames,
                      local_matrices_column_major=[[matrices[(j*frames+f)*16:(j*frames+f+1)*16] for f in range(frames)] for j in range(joint_count)])
    original_indices = list(range(original_count)) + extra
    skin, offset = [], 0
    for count in counts:
        skin.append(dict(zip(indices[offset:offset+count], weights[offset:offset+count])))
        offset += count
    result.update(positions=[positions[i*3:i*3+3] for i in original_indices],
                  uv=[uv[i*2:i*2+2] for i in range(vertex_count)],
                  original_indices=original_indices,
                  weights=[skin[i] for i in original_indices] if skin else [],
                  faces=[faces[i:i+3] for i in range(0, len(faces), 3)])
    return result
