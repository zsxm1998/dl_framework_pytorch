#copy from https://github.com/makinacorpus/easydict and modified.
class EasyDict(dict):
    """
    Get attributes

    >>> d = EasyDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'bar'

    Works recursively

    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1

    Bullet-proof

    >>> EasyDict({})
    {}
    >>> EasyDict(d={})
    {}
    >>> EasyDict(None)
    {}
    >>> d = {'a': 1}
    >>> EasyDict(**d)
    {'a': 1}

    Set attributes

    >>> d = EasyDict()
    >>> d.foo = 3
    >>> d.foo
    3
    >>> d.bar = {'prop': 'value'}
    >>> d.bar.prop
    'value'
    >>> d
    {'foo': 3, 'bar': {'prop': 'value'}}
    >>> d.bar.prop = 'newer'
    >>> d.bar.prop
    'newer'


    Values extraction

    >>> d = EasyDict({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
    >>> isinstance(d.bar, list)
    True
    >>> from operator import attrgetter
    >>> map(attrgetter('x'), d.bar)
    [1, 3]
    >>> map(attrgetter('y'), d.bar)
    [2, 4]
    >>> d = EasyDict()
    >>> d.keys()
    []
    >>> d = EasyDict(foo=3, bar=dict(x=1, y=2))
    >>> d.foo
    3
    >>> d.bar.x
    1

    Still like a dict though

    >>> o = EasyDict({'clean':True})
    >>> o.items()
    [('clean', True)]

    And like a class

    >>> class Flower(EasyDict):
    ...     power = 1
    ...
    >>> f = Flower()
    >>> f.power
    1
    >>> f = Flower({'height': 12})
    >>> f.height
    12
    >>> f['power']
    1
    >>> sorted(f.keys())
    ['height', 'power']

    update and pop items
    >>> d = EasyDict(a=1, b='2')
    >>> e = EasyDict(c=3.0, a=9.0)
    >>> d.update(e)
    >>> d.c
    3.0
    >>> d['c']
    3.0
    >>> d.get('c')
    3.0
    >>> d.update(a=4, b=4)
    >>> d.b
    4
    >>> d.pop('a')
    4
    >>> d.a
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'a'
    """
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in {'update', 'pop', 'convert2dict'}:
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        def recur_cvt_lt(value):
            dsttype = value.__class__
            dst = []
            for x in value:
                if isinstance(x, dict):
                    dst.append(self.__class__(x))
                elif isinstance(x, (list, tuple)):
                    dst.append(recur_cvt_lt(x))
                else:
                    dst.append(x)
            return dsttype(dst)
        
        if isinstance(value, (list, tuple)):
            value = recur_cvt_lt(value)
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)
    
    def convert2dict(self):
        def recur_cvt_lt(value):
            dsttype = value.__class__
            dst = []
            for x in value:
                if isinstance(x, self.__class__):
                    dst.append(x.convert2dict())
                elif isinstance(x, (list, tuple)):
                    dst.append(recur_cvt_lt(x))
                else:
                    dst.append(x)
            return dsttype(dst)
        
        d = {}
        for k, v in self.__dict__.items():
            if not (k.startswith('__') and k.endswith('__')) and not k in {'update', 'pop', 'convert2dict'}:
                if isinstance(v, self.__class__):
                    v = v.convert2dict()
                elif isinstance(v, (list, tuple)):
                    v = recur_cvt_lt(v)
                d[k] = v
        return d

    
if __name__ == '__main__':
    # 该类不能转换value为list或tuple中的list或tuple元素中的dict为EasyDict
    # 修改后可以了
    tx = {'k': [(1, {'kk': 2}), 1], 'ezd': {'key':'value'}}
    eztx = EasyDict(tx)
    x = eztx.k[0][1]
    print(x, type(x)) #{'kk': 2} <class 'dict'>