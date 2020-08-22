from struct import pack, unpack, calcsize, Struct

import numpy

def _group(fields, values):
    out = []
    i = 0

    for (name, fmt) in fields:
        n = len(fmt)
        if n > 1:
            out.append((name, fmt, values[i:i + n]))
            i += n
        else:
            out.append((name, fmt, values[i]))
            i += 1

    return out

def _mul(l):
    p = 1
    for o in l:
        p *= o
    return p

def _read_one(f, fmt):
    return unpack('<' + fmt, f.read(calcsize(fmt)))[0]

def _write_one(f, fmt, value):
    return f.write(pack('<' + fmt, value))

class FixedFieldsParser:
    def __init__(self, fields):
        self._fields = fields
        self._parser = Struct('<' + ''.join([fmt for (name, fmt) in self.fields]))

    def read(self, f, obj=None):
        obj = {} if obj is None else obj

        chunk = self.parser.unpack(f.read(self.parser.size))
        obj.update({name: value for (name, fmt, value) in _group(self.fields, chunk)})

        return obj

    def write(self, f, obj):
        chunk = self.parser.pack(*[obj[name] for (name, fmt) in self.fields])
        f.write(chunk)

    @property
    def parser(self):
        return self._parser

    @property
    def fields(self):
        return self._fields

class VariableStringParser:
    def __init__(self, name, fmt):
        self._name = name
        self._fmt = fmt

    def read(self, f, obj=None):
        obj = {} if obj is None else obj

        n = _read_one(f, self._fmt)
        obj[self._name] = f.read(n).decode('ascii')

        return obj

    def write(self, f, obj):
        _write_one(f, self._fmt, len(obj[self._name]))
        f.write(obj[self._name].encode('ascii'))

class ArrayParser:
    def __init__(self, name, shape_names, dtype):
        self._name = name
        self._shape_names = shape_names
        self._dtype = dtype

    def read(self, f, obj):
        shape = [obj[k] for k in self._shape_names]
        obj[self._name] = numpy.fromfile(f, dtype=self._dtype, count=_mul(shape)).reshape(shape)

    def write(self, f, obj):
        shape = [obj[k] for k in self._shape_names]

        arr = numpy.array(obj[self._name], dtype=self._dtype)
        if arr.shape != shape:
            arr = numpy.squeeze(arr)
        if arr.shape != shape:
            raise RuntimeError(f'array shape does not match expected: {arr.shape} vs {shape}')

        f.write(arr.tobytes())
