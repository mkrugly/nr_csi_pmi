from dataclasses import dataclass
import logging
import math
import numpy as np
import numpy.typing as npt
from operator import itemgetter
import pandas as pd
import random
from typing import Any, TypeVar

if __name__ == '__main__':
    from nr_csi_pmi.base import Base, BaseMatrix, PcsirsN1N2, I11, I12, Vm1m2
else:
    from .base import Base, BaseMatrix, PcsirsN1N2, I11, I12, Vm1m2


logger = logging.getLogger(__name__)


class K1P1:
    """
    38.214 Table 5.2.2.2.3-2

    for mapping i_14l
    """
    tab: list[float] = [
        0,
        0.125,
        (1/32)**0.5,
        0.25,
        0.125**0.5,
        0.5,
        0.5**0.5,
        1
    ]

    @classmethod
    def p1(cls, k1: int) -> float:
        return cls.tab[k1] if 0 <= k1 < len(cls.tab) else 0


class K2P2:
    """
    38.214 Table 5.2.2.2.3-3

    for mapping i_22l
    """
    tab: list[float] = [
        0.5**0.5,
        1
    ]

    @classmethod
    def p2(cls, k2: int) -> float:
        return cls.tab[k2] if 0 <= k2 < len(cls.tab) else 0


class I13l(Base):
    def __init__(self, i13l: int | str, L: int = 2):
        super().__init__()
        self._L = L
        self._size = self.get_bit_width(L)
        self.as_bin = i13l

    @classmethod
    def get_bit_width(cls, L: int) -> int:
        return cls.bit_size_log(2 * L)


class I14l(Base):
    _size_l: int = 3
    def __init__(self, i14l: int | str | list[int], L: int = 2):
        super().__init__()
        self._L = L
        self._num: int = 2 * L - 1
        self._size =  self.get_bit_width(L=L)
        self._list: npt.NDArray[int] = np.zeros(self._num, dtype=int)
        self._init(i14l=i14l)

    def _init(self, i14l: int | str | list[int|str]):
        if isinstance(i14l, list):
            i14l = "".join([self.to_bin_str(v=i, size=self._size_l) for i in i14l])
        self.as_bin = i14l
        lst = []
        for i in range(self._num):
            l = i + 1
            v = int(self.as_bin[i * self._size_l: l * self._size_l], 2)
            lst.append(v)
        self._list = np.array(lst)

    @property
    def M_l(self) -> int:
        return int(np.count_nonzero(self._list)) + 1

    @property
    def as_list(self) -> list[int]:
        return self._list.tolist()

    @property
    def strongest(self) -> list[int]:
        self._li

    @classmethod
    def get_bit_width(cls, L: int) -> int:
        return cls._size_l * (2 * L - 1)


class I21l(Base):
    def __init__(self, i21l: int | str | list[int], M_l: int, sbAmp: bool = True, Npsk: int = 8, L: int = 2):
        super().__init__()
        self._L = L
        self._M_l: int = M_l
        self._sbAmp: bool = sbAmp
        self._Npsk: int = Npsk
        self._num_l: int = min(self.K2_for_l(L), M_l) - 1 if sbAmp else M_l - 1
        self._num_s: int = M_l - (self._num_l + 1)
        self._size_l: int = self.bit_size_log(Npsk)
        self._size =  self.get_bit_width(M_l=M_l, Npsk=Npsk, sbAmp=sbAmp, L=L)
        self._list: npt.NDArray[int] = np.zeros(self._num_l + self._num_s, dtype=int)
        self._init(i21l=i21l)

    def _init(self, i21l: int | str | list[int|str]):
        if isinstance(i21l, list):
            i21l = "".join([self.to_bin_str(v=v, size=self._size_l if i < self._num_l else 2) for i, v in enumerate(i21l)])
        self.as_bin = i21l
        lst = []
        for i in range(self._num_l + self._num_s):
            l = i + 1
            _size = self._size_l if i < self._num_l else 2
            v = int(self.as_bin[i * _size: l * _size], 2)
            lst.append(v)
        self._list = np.array(lst)

    @property
    def M_l(self) -> int:
        return self._M_l

    @property
    def min_K2_M_l(self) -> int:
        return self._num_l

    @property
    def num_s(self) -> int:
        return self._num_s

    @property
    def as_list(self) -> list[int]:
        return self._list.tolist()

    @classmethod
    def get_bit_width(cls, M_l: int, Npsk: int, sbAmp: bool, L: int) -> int:
        if M_l <= 0:
            return 0
        _size_l = cls.bit_size_log(Npsk)
        if sbAmp:
            _min = min(cls.K2_for_l(L), M_l)
            return (_min - 1) * _size_l + 2 * (M_l - _min)
        else:
            return (M_l - 1) * _size_l

    @classmethod
    def K2_for_l(cls, L: int) -> int:
        return 6 if L==4 else 4


class I22l(Base):
    _size_l: int = 1
    def __init__(self, i22l: int | str | list[int], M_l: int, sbAmp: bool = True, L: int = 2):
        super().__init__()
        self._L = L
        self._M_l: int = M_l
        self._sbAmp: bool = sbAmp
        self._size: int =  self.get_bit_width(M_l=M_l, sbAmp=sbAmp, L=L)
        self._num_l: int = self._size
        self._list: npt.NDArray[int] = np.zeros(self._num_l, dtype=int)
        self._init(i22l=i22l)

    def _init(self, i22l: int | str | list[int|str]):
        if self._sbAmp:
            if isinstance(i22l, list):
                i22l = "".join([self.to_bin_str(v=v, size=self._size_l) for v in i22l])
            self.as_bin = i22l
            lst = []
            for i in range(self._num_l):
                l = i + 1
                v = int(self.as_bin[i * self._size_l: l * self._size_l], 2)
                lst.append(v)
            self._list = np.array(lst)

    @property
    def M_l(self) -> int:
        return self._M_l

    @property
    def as_list(self) -> list[int]:
        return self._list.tolist()

    @classmethod
    def get_bit_width(cls, M_l: int, sbAmp: bool, L: int) -> int:
        return min(cls.K2_for_l(L), M_l) - 1 if sbAmp and M_l > 0 else 0

    @classmethod
    def K2_for_l(cls, L: int) -> int:
        return 6 if L==4 else 4


GenXType = TypeVar('GenXType', I13l, I14l, I22l, I21l)

class BaseX:
    def __init__(self, L: int = 2):
        self._list: list[GenXType] = []
        self._L: int = L

    def _for_l(self, l: int = 1) -> GenXType | None:
        return self._list[l-1] if l <= len(self._list) else None

    @property
    def v(self) -> int:
        return len(self._list)

    @property
    def L(self) -> int:
        return self._L

    @property
    def as_list(self) -> list[GenXType]:
        return self._list

    @property
    def as_bin_list(self) -> list[str]:
        return [i.as_bin for i in self._list]

    @property
    def as_int_list(self) -> list[int]:
        return [i.as_int for i in self._list]

    @property
    def as_list_tuple(self) -> tuple[list[int],...]:
        return tuple(i.as_list for i in self._list if hasattr(i, "as_list"))


class I13(BaseX):
    def __init__(self, i13: tuple[int | str | I13l, ...], L: int = 2):
        super().__init__(L=L)
        self._init(i13)

    def _init(self, i13: tuple[int | str | list[int] | I13l, ...]):
        for i13l in i13:
            if i13l is not None:
                self._list.append(i13l if isinstance(i13l, I13l) else I13l(i13l=i13l, L=self._L))

    @property
    def i131(self) -> I13l | None:
        return self._for_l(1)

    @property
    def i132(self) -> I13l | None:
        return self._for_l(2)


class I14(BaseX):
    def __init__(self, i14: tuple[int | str | list[int] | I14l, ...], L: int = 2):
        super().__init__(L=L)
        self._L = L
        self._init(i14)

    def _init(self, i14: tuple[int | str | list[int] | I14l, ...]):
        for i14l in i14:
            if i14l is not None:
                self._list.append(i14l if isinstance(i14l, I14l) else I14l(i14l=i14l, L=self._L))

    @property
    def M_1(self) -> int:
        return self._for_l(1).M_l if self._for_l(1) else 0

    @property
    def M_2(self) -> int:
        return self._for_l(2).M_l if self._for_l(2) else 0

    @property
    def i141(self) -> I14l | None:
        return self._for_l(1)

    @property
    def i142(self) -> I14l | None:
        return self._for_l(2)


class I21(BaseX):
    def __init__(self, i21: tuple[int | str | list[int] | I21l, ...], M_l: tuple[int, ...],
                 sbAmp: bool = True, Npsk: int = 8, L: int = 2):
        super().__init__(L=L)
        self._L = L
        self._M_l: tuple[int, ...] = M_l
        self.sbAmp: bool = sbAmp
        self.Npsk: int = Npsk
        self._init(i21)

    def _init(self, i21: tuple[int | str | list[int] | I21l, ...]):
        for i, v in enumerate(i21):
            if v:
                self._list.append(v if isinstance(v, I21l)
                                  else I21l(i21l=v, M_l=self._M_l[i], sbAmp=self.sbAmp, Npsk=self.Npsk, L=self._L)
                                  )

    @property
    def i211(self) -> I21l | None:
        return self._for_l(1)

    @property
    def i212(self) -> I21l | None:
        return self._for_l(2)

    @property
    def i211_as_int(self) -> int | None:
        return self.i211.as_int if self.i211 is not None else None

    @property
    def i212_as_int(self) -> int | None:
        return self.i212.as_int if self.i212 is not None else None

    @property
    def min_K2_M_l(self) -> tuple[int,...]:
        return (self.i211.min_K2_M_l, self.i212.min_K2_M_l) if self.i212 else (self.i211.min_K2_M_l,)

class I22(BaseX):
    def __init__(self, i22: tuple[int | str | list[int] | I22l, ...], M_l: tuple[int, ...],
                 sbAmp: bool = True, L: int = 2):
        super().__init__(L=L)
        self._L = L
        self.M_l: tuple[int, ...] = M_l
        self.sbAmp: bool = sbAmp
        self._init(i22)

    def _init(self, i22: tuple[int | str | list[int] | I22l, ...]):
        for i, v in enumerate(i22):
            if v:
                self._list.append(v if isinstance(v, I22l)
                                  else I22l(i22l=v, M_l=self.M_l[i], sbAmp=self.sbAmp, L=self._L)
                                  )

    @property
    def i221(self) -> I22l | None:
        return self._for_l(1)

    @property
    def i222(self) -> I22l | None:
        return self._for_l(2)

    @property
    def i221_as_int(self) -> int | None:
        return self.i221.as_int if self.i221 is not None else None

    @property
    def i222_as_int(self) -> int | None:
        return self.i222.as_int if self.i222 is not None else None


@dataclass
class X1:
    i11: I11
    i12: I12
    i13: I13
    i14: I14

    @classmethod
    def get_sizes(cls, N1N2: tuple[int, int], O1O2: tuple[int, int], L: int = 2, v: int = 1) -> tuple[int,...]:
        v1 = I11.get_bit_width(O1O2), I12.get_bit_width(N1N2, L), I13l.get_bit_width(L), I14l.get_bit_width(L)
        v2 = v1 + (I13l.get_bit_width(L), I14l.get_bit_width(L))
        return v1 if v == 1 else v2

    @classmethod
    def from_str(cls, x: str, N1N2: tuple[int, int], O1O2: tuple[int, int], L: int = 2, v: int = 1):
        sizes = cls.get_sizes(N1N2,O1O2, L, v)  # sizes for i11, i12, i13, i14
        x1_size = sum(sizes)
        x = Base.to_bin_str(v=x, size=x1_size)
        assert len(x) == x1_size, f"wrong x1 bitwidth: {len(x)} (allowed values {x1_size} if v=={v})"

        offset = 0
        x1_split = []
        for i in sizes:
            x1_split.append(x[offset:offset+i])
            offset += i
        i11 = I11(i11_or_q1q2=x1_split[0], O1O2=O1O2)
        i12 = I12(i12_or_n1n2=x1_split[1], N1N2=N1N2, L=L)
        i13 = (x1_split[2], x1_split[4] if len(x1_split) > 4 else None)
        i14 = (x1_split[3], x1_split[5] if len(x1_split) > 4 else None)
        return cls(i11=i11, i12=i12, i13=I13(i13=i13, L=L), i14=I14(i14=i14, L=L))

    @classmethod
    def from_beam(cls, n1n2: tuple[list[int], list[int]], q1q2: tuple[int, int],
                  strongest: tuple[int, ...], k1: tuple[list[int], ...],
                  N1N2: tuple[int, int], O1O2: tuple[int, int], L: int = 2):
        i11 = I11(i11_or_q1q2=q1q2, O1O2=O1O2)
        i12 = I12(i12_or_n1n2=n1n2, N1N2=N1N2, L=L)
        i13 = I13(i13=strongest, L=L)
        i14 = I14(i14=k1, L=L)
        return cls(i11=i11, i12=i12, i13=i13, i14=i14)

    @property
    def as_bin(self) -> str:
        i13_14_str = ''.join([''.join(i) for i in zip(self.i13.as_bin_list, self.i14.as_bin_list)])
        return f"{self.i11.as_bin}{self.i12.as_bin}{i13_14_str}"

    @property
    def as_hex(self) -> str:
        return hex(Base.align_bit_length(v=int(self.as_bin, 2), size=len(self.as_bin), factor=4))

    @property
    def strongest(self) -> list[int]:
        return self.i13.as_int_list

    @property
    def strongest_flat(self) -> list[int]:
        return [2*self.i13.L*i + v for i, v in enumerate(self.strongest)]

    @property
    def other(self) -> list[int]:
        return self.i13.as_int_list

    @property
    def other_flat(self) -> list[int]:
        return [2*self.i13.L*i + v for i, v in enumerate(self.strongest)]


@dataclass
class X2sb:
    subband: int
    sbAmp: bool
    i21: I21
    i22: I22

    @classmethod
    def get_sizes(cls, M_l: tuple[int,...], Npsk: int, sbAmp: bool, L: int = 2) -> tuple[int, ...]:
        sizes = []
        for m_l in M_l:
            sizes.append(I21l.get_bit_width(M_l=m_l, Npsk=Npsk, sbAmp=sbAmp, L=L))
        for m_l in M_l:
            sizes.append(I22l.get_bit_width(M_l=m_l, sbAmp=sbAmp, L=L))
        return tuple(sizes)

    @classmethod
    def from_str(cls, x: str, subband: int, M_l: tuple[int,...], Npsk: int, sbAmp: bool, L: int = 2):
        sizes = cls.get_sizes(M_l=M_l,Npsk=Npsk, sbAmp=sbAmp, L=L)
        v = len(M_l)
        str_lst = []
        offset = 0
        for i in sizes:
            str_lst.append(x[offset:offset+i])
            offset += i
        i21=I21(i21=tuple(str_lst[0:v]), M_l=M_l, Npsk=Npsk, sbAmp=sbAmp, L=L)
        i22=I22(i22=tuple(str_lst[v:v+v]), M_l=M_l, sbAmp=sbAmp, L=L)
        return cls(subband=subband, i21=i21, i22=i22, sbAmp=sbAmp)

    @classmethod
    def from_beam(cls, k2: tuple[list[int],...], c: tuple[list[int],...], subband: int,
                  Npsk: int, sbAmp: bool, L: int = 2):
        M_l = tuple(len(i)+1 for i in c)
        k2 = k2 or tuple()
        i21=I21(i21=c, M_l=M_l, Npsk=Npsk, sbAmp=sbAmp, L=L)
        i22=I22(i22=k2, M_l=M_l, sbAmp=sbAmp, L=L)
        return cls(subband=subband, i21=i21, i22=i22, sbAmp=sbAmp)

    @property
    def as_bin(self) -> str:
        return ''.join(self.i21.as_bin_list) + ''.join(self.i22.as_bin_list)

    @property
    def as_dict(self) -> dict[str,int|None]:
        return {
            "sb": self.subband,
            "i211": self.i21.i211_as_int,
            "i212": self.i21.i212_as_int,
            "i221": self.i22.i221_as_int,
            "i222": self.i22.i222_as_int,
        }

    @property
    def c(self) -> tuple[list[int],...]:
        return self.i21.as_list_tuple

    @property
    def k2(self) -> tuple[list[int],...]:
        return self.i22.as_list_tuple


class X2:
    def __init__(self, x2: list[X2sb]):
        self._list: list[X2sb] = sorted(x2, key=lambda x: x.subband) # ensure the list is sorted by subband index
        self._sbinx_inx_map: dict[int, int] = {}
        self._init()

    def _init(self):
        self._sbinx_inx_map.update({v.subband: i for i, v in enumerate(self._list)})

    @property
    def num_sb(self) -> int:
        return len(self._list)

    @property
    def as_bin(self) -> str:
        return self.as_bin_even + self.as_bin_odd

    @property
    def as_hex(self) -> str:
        return hex(Base.align_bit_length(v=int(self.as_bin, 2), size=len(self.as_bin), factor=4))

    @property
    def as_bin_odd(self) -> str:
        return "".join([i.as_bin for i in self.as_list_odd])

    @property
    def as_bin_even(self) -> str:
        return "".join([i.as_bin for i in self.as_list_even])

    @property
    def as_bin_flat(self) -> str:
        return "".join([i.as_bin for i in self.as_list])

    @property
    def as_list(self) -> list[X2sb]:
        return self._list

    @property
    def as_list_odd(self) -> list[X2sb]:
        return [i for i in self.as_list if i.subband % 2]

    @property
    def as_list_even(self) -> list[X2sb]:
        return [i for i in self.as_list if not i.subband % 2]

    @property
    def as_dict_list(self) -> list[dict[str,int|None]]:
        return [i.as_dict for i in self.as_list]

    @property
    def c(self) -> list[tuple[list[int],...]]:
        return [i.c for i in self.as_list]

    @property
    def k2(self) -> list[tuple[list[int],...]]:
        return [i.k2 for i in self.as_list]

    @property
    def subbands(self) -> list[int]:
        return [i.subband for i in self.as_list]

    def for_sb(self, subband: int) -> X2sb | None:
        inx = self._sbinx_inx_map.get(subband, -1)
        return self._list[inx] if -1 < inx < len(self._list) else None

    @classmethod
    def from_str(cls, x: str, subbands: list[int], M_l: tuple[int,...],
                 Npsk: int, sbAmp: bool, L: int = 2):
        size_l = sum(X2sb.get_sizes(M_l=M_l,Npsk=Npsk, sbAmp=sbAmp, L=L))
        x = Base.to_bin_str(v=x, size=size_l*len(subbands))
        offset = 0

        def _get_x2sb(_subband: int) -> X2sb:
            nonlocal offset
            _end = offset + size_l
            _x = x[offset:_end]
            offset += size_l
            return X2sb.from_str(x=_x, subband=_subband, M_l=M_l,Npsk=Npsk, sbAmp=sbAmp, L=L)
        _even_odd_subbands = [i for i in sorted(subbands) if not i % 2] + [i for i in sorted(subbands) if i % 2]
        return cls([_get_x2sb(_subband=i) for i in _even_odd_subbands])

    @classmethod
    def from_beam(cls, k2: list[tuple[list[int],...]], c: list[tuple[list[int],...]],
                  subbands: list[int], Npsk: int, sbAmp: bool, L: int = 2):
        k2 = k2 or [()] * len(subbands)
        def _get_x2sb(i: int, _subband: int) -> X2sb:
            return X2sb.from_beam(k2=k2[i], c=c[i], subband=_subband, Npsk=Npsk, sbAmp=sbAmp, L=L)

        return cls([_get_x2sb(i, v) for i, v in enumerate(subbands)])


class P1(BaseMatrix):
    def __init__(self, x1: X1):
        self.i13: I13 = x1.i13
        self.i14: I14 = x1.i14
        # matrix of shape (v, 2L)
        super().__init__(shape=(self.i13.v, 2 * self.i13.L), dtype=np.float64, vsplit=self.i13.v,
                         shape_desc="v, 2L")
        self._init()

    def _init(self):
        np.put(self.matrix, self.strongest_flat, K1P1.p1(7))
        self.matrix[self.matrix==0] = [K1P1.p1(i) for i in self.k1_flat]

    def __str__(self):
        return (f"{self.__class__}: strongest:\n{self.strongest}\n"
                f"matrix {self.shape}:\n{self.matrix}")

    @property
    def strongest(self) -> list[int]:
        return self.i13.as_int_list

    @property
    def strongest_flat(self) -> list[int]:
        return [2*self.i13.L*i + v for i, v in enumerate(self.strongest)]

    @property
    def other(self) -> list[int]:
        return self.i13.as_int_list

    @property
    def k1(self) -> tuple[list[int],...]:
        return self.i14.as_list_tuple

    @property
    def k1_flat(self) -> npt.NDArray[int]:
        return np.array(self.k1, dtype=int).flatten()

    def for_l(self, l: int) -> npt.NDArray:
        return self.matrix[l - 1]


class BaseSb(BaseMatrix):
    def __init__(self, x1: X1, x2: X2, dtype: npt.DTypeLike):
        self.i13: I13 = x1.i13
        self.i14: I14 = x1.i14
        self.x2: X2 = x2
        self.num_sb: int = self.x2.num_sb
        self.v: int  = self.i14.v
        # matrix of shape (v*numSb, 2L)
        super().__init__((self.v * self.num_sb, 2 * self.i14.L), dtype=dtype,
                         vsplit=self.v, shape_desc="v*numSb, 2L")

    def __str__(self):
        return f"{self.__class__}: matrix {self.shape} ({self.shape_desc}):\n{self.matrix}"

    @property
    def strongest_pos(self) -> list[int]:
        # from reported i14, get positions of strongest beams sorted in descending order
        _sorted_i14_pos = np.argsort(-np.asarray(self.i14.as_list_tuple))
        # convert i13 list to column vector (to match the layer dimension in i14 array
        _i13_pos_vec = np.c_[self.i13.as_int_list]
        # increment all i14 positions >= i13 to get the absolute position (considering the strongest, not reported beam)
        return np.add(_sorted_i14_pos, 1, out=_sorted_i14_pos, where=_sorted_i14_pos >= _i13_pos_vec).tolist()


class P2(BaseSb):
    def __init__(self, x1: X1, x2: X2):
        super().__init__(x1=x1, x2=x2, dtype=np.float64)
        self._init()

    def _init(self):
        # set all values to 1
        self.matrix[:] = K2P2.p2(1)
        for s, x2sb in enumerate(self.x2.as_list):
            if not x2sb.sbAmp:
                break
            for l, k2l in enumerate(x2sb.k2):
                _pos_flat = np.sort(np.array(self.strongest_pos[l][:len(k2l)])) + self.shape[1]*(s+l*self.num_sb)
                _p2 = [K2P2.p2(i) for i in k2l]
                assert len(_p2) > 0, f"Missing subband parameters: k_2/p_2 not found (sb#{x2sb.subband}, sbAmp:{x2sb.sbAmp})"
                np.put(self.matrix, _pos_flat, _p2)


    @property
    def k2(self) -> list[tuple[list[int],...]]:
        return self.x2.k2


class Phi(BaseSb):
    def __init__(self, x1: X1, x2: X2):
        super().__init__(x1=x1, x2=x2, dtype=np.complex128)
        self._init()

    def _init(self):
        # set all values to 1
        self.matrix[:] = 1
        for s, x2sb in enumerate(self.x2.as_list):
            for l, c_phil in enumerate(x2sb.c):
                _pos_flat = np.sort(np.array(self.strongest_pos[l][:len(c_phil)])) + self.shape[1] * (s + l * self.num_sb)
                _num_l = x2sb.i21.min_K2_M_l[l]
                _npsk = x2sb.i21.Npsk
                _phi = [np.around(np.exp((1j * 2 * np.pi * c) / (_npsk if i < _num_l else 4)), 3)
                        for i, c in enumerate(c_phil)]
                np.put(self.matrix, _pos_flat, _phi)

    @property
    def c(self) -> list[tuple[list[int],...]]:
        return self.x2.c


class P1P2Phi(BaseMatrix):
    def __init__(self, p1: P1, p2: P2, phi:Phi):
        self.p1: P1 = p1
        self.p2: P2 = p2
        self.phi: Phi = phi
        # matrix of shape (v*numSb, 2L)
        super().__init__(self.phi.shape, dtype=np.complex128,
                         vsplit=self.p1.shape[0], shape_desc="v*numSb, 2L")
        self._gamma: npt.NDArray = np.zeros(self.shape[0], dtype=np.float64)
        self._init()

    def _init(self):
        # each row of the p1 matrix (representing p1 values for a certain layer)
        # shall be repeated num_sb number of times (i.e. to be applied for all subbands)
        num_rep = int(self.shape[0] / self.p1.shape[0])
        p1_repeated = np.repeat(self.p1.matrix, num_rep, axis=0)
        p1_repeated_x_p2 = np.multiply(p1_repeated, self.p2.matrix)
        self._matrix = np.multiply(p1_repeated_x_p2, self.phi.matrix)
        _matrix_2 = np.square(p1_repeated_x_p2)
        self._gamma = np.sum(_matrix_2, axis=1)

    def for_l(self, l: int) -> npt.NDArray:
        assert 0 < l <= len(self.p1.matrix)
        return self.matrices[l - 1]

    @property
    def gamma_matrix(self) -> npt.NDArray:
        return self._gamma

    @property
    def gamma_matrices(self) -> list[npt.NDArray]:
        return np.vsplit(self.gamma_matrix, self._vsplit)

    def gamma_for_l(self, l: int) -> npt.NDArray:
        assert 0 < l <= len(self.p1.matrix)
        return self.gamma_matrices[l - 1]


class W(BaseMatrix):
    def __init__(self, vm1m2: Vm1m2, p1p2phi: P1P2Phi):
        self.vm1m2: Vm1m2 = vm1m2
        self.p1p2phi: P1P2Phi = p1p2phi
        self._v: int = self.p1p2phi.vsplit
        self._num_sb: int = int(self.p1p2phi.shape[0] / self._v)
        # matrix of shape (2*N1*N2, num_sb*v)
        super().__init__(shape=(self.vm1m2.vlm_w1.shape[0], self.p1p2phi.shape[0]),
                         dtype=np.complex128, shape_desc="2*N1*N2, num_sb*v")

        self._init()

    def _init(self):
        # Vm1m2 is common for all layers and has a shape 2N1N2 x 2L
        # P1P2Phi includes coefficients for all subbands and all layers and has a shape v*num_sb x 2L
        # therefore for the matrix multiplication, the P1P2Phi has to be transposed
        # (this way we will get the precoding vectors for all subbands and layers in columns
        _factor = (1 / np.sqrt(self.N1 * self.N2 * self.p1p2phi.gamma_matrix))
        self._matrix = _factor * np.matmul(self.vm1m2.vlm_w1, self.p1p2phi.matrix_T)
        # additional scaling for v > 1 - see 38.214 Table 5.2.2.2.3-5
        self._matrix *= self._get_w_scaling()

    def _get_w_scaling(self) -> float:
        _lst = (1, 1/2**0.5)
        return _lst[self._v - 1] if 0 < self._v <= len(_lst) else 1

    @property
    def num_sb(self) -> int:
        return self._num_sb

    @property
    def N1(self) -> int:
        return self.vm1m2.i12.N1

    @property
    def N2(self) -> int:
        return self.vm1m2.i12.N2

    def for_sb(self, inx: int) -> npt.NDArray:
        assert 0 <= inx < self.num_sb, f"Wrong subband (values 0..{self.num_sb - 1} allowed)"
        return self.matrix[:, inx::self.num_sb]

    def for_layer(self, l: int) -> npt.NDArray:
        assert 0 < l <= self._v, f"Wrong layer (values 1..{self._v} allowed)"
        start = (l - 1) * self.num_sb
        end = start + self.num_sb
        return self.matrix[:, start:end]


@dataclass
class Pmi:
    w: W
    x1: X1
    x2: X2

    @property
    def vm1m2(self) -> Vm1m2:
        return self.w.vm1m2

    @property
    def p1p2phi(self) -> P1P2Phi:
        return self.w.p1p2phi

    @property
    def p1(self) -> P1:
        return self.p1p2phi.p1

    @property
    def p2(self) -> P2:
        return self.p1p2phi.p2

    @property
    def phi(self) -> Phi:
        return self.p1p2phi.phi

    @property
    def coefs(self) -> dict[str, Any]:
        return {
            "i11": self.x1.i11.as_int,
            "i12": self.x1.i12.as_int,
            "i131": self.x1.i13.i131.as_int,
            "i141": self.x1.i14.i141.as_int,
            "i132": self.x1.i13.i132.as_int if self.x1.i13.i132 else -1,
            "i142": self.x1.i14.i142.as_int if self.x1.i14.i142 else -1,
            "x2": self.x2.as_dict_list,
        }

    @property
    def coefs_bin(self) -> dict[str, Any]:
        return {
            "i11": self.x1.i11.as_bin,
            "i12": self.x1.i12.as_bin,
            "i131": self.x1.i13.i131.as_bin,
            "i141": self.x1.i14.i141.as_bin,
            "i132": self.x1.i13.i132.as_bin if self.x1.i13.i132 else "",
            "i142": self.x1.i14.i142.as_bin if self.x1.i14.i142 else "",
            "x2": self.x2.as_bin,
        }

    @property
    def coefs_str(self) -> str:
        """
        """
        i1x2_str = f"i132 {self.x1.i13.i132.as_bin}, i142 {self.x1.i14.i142.as_bin}, " if self.x1.i13.i132 else ""
        return (f"M1 {self.x1.i14.M_1}, M2 {self.x1.i14.M_2} [i11 {self.x1.i11.as_bin}, i12 {self.x1.i12.as_bin}, "
                f"i131 {self.x1.i13.i131.as_bin}, i141 {self.x1.i14.i141.as_bin}, {i1x2_str}"
                f"X2 {self.x2.as_bin}]")

    @property
    def coefs_str_alt(self) -> str:
        """
        """
        return f"M1 {self.x1.i14.M_1} M2 {self.x1.i14.M_2} PMI [X1 {self.x1.as_bin}, X2 {self.x2.as_bin}]"

    @property
    def coefs_hex(self) -> str:
        """
        """
        return f"M1 {self.x1.i14.M_1} M2 {self.x1.i14.M_2} PMI [X1 {self.x1.as_hex}, X2 {self.x2.as_hex}]"

    @property
    def coefs_hex_alt(self) -> str:
        """
        """
        return self.coefs_hex.replace("0x", "").upper()

    @property
    def coefs_hex_part2(self) -> str:
        """
        """
        x = self.x1.as_bin + self.x2.as_bin
        return (f"M1 {self.x1.i14.M_1} M2 {self.x1.i14.M_2} X "
                f"{hex(Base.align_bit_length(v=int(x, 2), size=len(x), factor=4))}")

    @property
    def beams(self) -> dict[str, Any]:
        return {
            "q1q2": self.x1.i11.q1q2,
            "n1n2": self.x1.i12.n1n2,
            "strongest": self.x1.i13.as_int_list,
            "k1": self.x1.i14.as_list_tuple,
            "k2": self.x2.k2,
            "c": self.x2.c,
            "subbands": self.x2.subbands,
        }

    def log_verbose(self):
        logger.info(self.vm1m2)
        logger.info(self.w)
        logger.info(self.p1)
        logger.info(self.p2)
        logger.info(self.phi)
        logger.info(self.p1p2phi)
        logger.info(f"Gamma matrix:\n{self.p1p2phi.gamma_matrix}")
        logger.info(self.w)

    def log_summary(self):
        logger.info(f"Coefficients: {self.coefs}")
        logger.info(f"Beam settings: {self.beams}")
        logger.info(f"Coefficients bin: {self.coefs_bin}")
        logger.info(f"Coefficients hex: {self.coefs_hex_alt}")
        logger.info(f"Coefficients part2 hex: {self.coefs_hex_part2}")


class PmiGenerator:
    def __init__(self, N1N2: tuple[int, int], subbands: list[int]|str, Npsk: int, sbAmp: bool, L: int = 2, **kwargs):
        self.subbands: list[int] = sorted(subbands) if isinstance(subbands, list) \
            else np.flatnonzero(np.array([int(i) for i in reversed(subbands)])).tolist()
        self.Npsk: int = Npsk
        self.sbAmp: bool = sbAmp
        self.L: int = L
        self.N1N2: tuple[int, int] = N1N2
        self.O1O2: tuple[int, int] = PcsirsN1N2.get_O1O2(self.N1N2)
        self.Pcsirs: int = PcsirsN1N2.get_Pcsirs(self.N1N2)

    def factory(self, x1: X1, x2: X2) -> Pmi:
        vm1m2 = Vm1m2(i11=x1.i11.as_int, i12=x1.i11.as_int, N1N2=self.N1N2, O1O2=self.O1O2, L=self.L, rel16=False)
        p1 = P1(x1=x1)
        p2 = P2(x1=x1, x2=x2)
        phi = Phi(x1=x1, x2=x2)
        p1p2phi = P1P2Phi(p1=p1, p2=p2, phi=phi)
        w = W(vm1m2=vm1m2, p1p2phi=p1p2phi)
        return Pmi(w=w, x1=x1, x2=x2)

    def beam_factory(self, n1n2: tuple[list[int], list[int]], q1q2: tuple[int, int],
                     strongest: tuple[int, ...], k1: tuple[list[int], ...],
                     k2: list[tuple[list[int],...]], c: list[tuple[list[int],...]], **kwargs) -> Pmi:
        _x1 = X1.from_beam(n1n2=n1n2, q1q2=q1q2, strongest=strongest, k1=k1, N1N2=self.N1N2, O1O2=self.O1O2, L=self.L)
        _x2 = X2.from_beam(k2=k2, c=c, subbands=self.subbands, Npsk=self.Npsk, sbAmp=self.sbAmp, L=self.L)
        return self.factory(x1=_x1, x2=_x2)

    def coef_factory(self, m_l: tuple[int,int], x1: str, x2: str, v:int, **kwargs) -> Pmi:
        _x1 = X1.from_str(x=x1, N1N2=self.N1N2, O1O2=self.O1O2, L=2, v=v)
        _x2 = X2.from_str(x=x2, subbands=self.subbands, M_l=m_l, Npsk=self.Npsk, sbAmp=self.sbAmp, L=self.L)
        return self.factory(x1=_x1, x2=_x2)

    def coef_part2_factory(self, m_l: tuple[int,int], x: str, v:int, **kwargs) -> Pmi:
        offset = sum(X1.get_sizes(N1N2=self.N1N2, O1O2=self.O1O2, L=2, v=v))
        _size = offset + len(self.subbands) * sum(X2sb.get_sizes(M_l=m_l,Npsk=self.Npsk, sbAmp=self.sbAmp, L=self.L))
        x = Base.to_bin_str(x, size=_size)

        x1 = X1.from_str(x=x[:offset], N1N2=self.N1N2, O1O2=self.O1O2, L=2, v=v)
        x2 = X2.from_str(x=x[offset:], subbands=self.subbands, M_l=m_l, Npsk=self.Npsk, sbAmp=self.sbAmp, L=self.L)
        return self.factory(x1=x1, x2=x2)


def example_coef_part2_factory_test(ant_dim: tuple[int, int], subbands: list[int]|str, x: str = "",
                                    m_l: tuple[int,int]|None = None, v: int = 2, Npsk: int=8, sbAmp: bool = True, L=2,
                                    optimize_rank: bool = True, randomize: bool = False, verbose: bool = False) -> Pmi:
    x = x or '111101111100000010100001010000101011000110110001101011111001001010011000100111011110100011000001100011100011000111100010001101110110001000010100001101010101000101001011001010011011111000001011101001101010111010110011110000001111100100101010110011011101011000101001111001010110010001001000011100100001010001110000010110100011101011100000100100111001100'
    m_l = m_l or (3, 4)
    gen = PmiGenerator(N1N2=ant_dim, subbands=subbands, Npsk=Npsk, sbAmp=sbAmp, L=L)
    pmi = gen.coef_part2_factory(m_l=m_l, x=x, v=v)
    if verbose:
        pmi.log_verbose()
    pmi.log_summary()
    return pmi


def example_coef_factory_test(ant_dim: tuple[int, int], subbands: list[int]|str, x1: str, x2: str, m_l: tuple[int,int],
                              v: int = 2, Npsk: int=8, sbAmp: bool = True, L=2, verbose: bool = False) -> Pmi:
    """Example function showcasing precoding matrix calculation for the given PMI coefficients
    """
    gen = PmiGenerator(N1N2=ant_dim, subbands=subbands, Npsk=Npsk, sbAmp=sbAmp, L=L)
    pmi = gen.coef_factory(m_l=m_l, x1=x1, x2=x2, v=v)
    if verbose:
        pmi.log_verbose()
    pmi.log_summary()
    return pmi


def example_pmi_generator(subbands: list[int]|str, Npsk: int=8, sbAmp: bool = True,
                          v: int = 2, optimize_rank: bool = True, verbose: bool = False) -> Pmi:
    # determine number of subbands
    num_sb = len(subbands) if isinstance(subbands, list) else subbands.count("1")
    strongest = {
        (1, True): (1,),
        (2, True): (1, 2),
        (2, False): (1, 1),
    }
    k1 = {
        (1, True): ([0, 1, 2],),
        (2, True): ([0, 1, 2], [2, 4, 1]),
        (2, False):([0, 1, 2], [0, 1, 2]),
    }
    k2_pattern = [([0, 1], [1, 0, 1]), ([1, 0], [1, 0, 1]), ([0, 1], [1, 0, 0]), ([1, 1], [1, 0, 0])]
    k2_size = len(k2_pattern)
    k2 = {
        (1, True): [(k2_pattern[i % k2_size][0],) for i in range(num_sb)],
        (2, True): [ k2_pattern[i % k2_size] for i in range(num_sb)],
        (2, False): [(k2_pattern[i % k2_size][0], k2_pattern[i % k2_size][0]) for i in range(num_sb)],
    }
    c_pattern = [([2, 6], [1, 5, 4]), ([6, 4], [6, 5, 3]), ([3, 7], [1, 1, 2]), ([4, 7], [4, 0, 3])]
    c_pattern = [tuple(list(map(lambda x: x % 4, i)) for i in t) for t in c_pattern] # make sure values correspond to Npsk
    c_size = len(c_pattern)
    c = {
        (1, True): [(c_pattern[i % c_size][0],) for i in range(num_sb)],
        (2, True): [ c_pattern[i % c_size] for i in range(num_sb)],
        (2, False): [(c_pattern[i % c_size][0], c_pattern[i % c_size][0]) for i in range(num_sb)],
    }

    ant_dim = (4, 2)
    L=2
    beam_settings = {'q1q2': (3, 3), 'n1n2': ([1, 3], [0, 1]),
                     'strongest': strongest.get((v, optimize_rank), strongest.get((1, True))),
                     'k1':  k1.get((v, optimize_rank), k1.get((1, True))),
                     'k2': k2.get((v, optimize_rank),  k2.get((1, True))) if sbAmp else [],
                     'c': c.get((v, optimize_rank), c.get((1, True))),
                     }

    gen = PmiGenerator(N1N2=ant_dim, subbands=subbands, Npsk=Npsk, sbAmp=sbAmp, L=L)
    pmi = gen.beam_factory(**beam_settings)
    if verbose:
        pmi.log_verbose()
    pmi.log_summary()
    return pmi


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    pmi = example_coef_part2_factory_test(ant_dim=(4, 2), subbands='01111111111111111',
                                    x="0xf7a0a942b1b1af92989de8c18e31e237621435514b29be0ba6aeb3c0f92acdd629e564487214705a3ae09398",
                                    v=2, L=2, Npsk=8, sbAmp=True)
    pmi = example_coef_factory_test(ant_dim=(4, 2), subbands='01111111111111111',
                                    x1="0xF7A0A942",
                                    x2="0x58D8D7C94C4EF460C718F11BB10A1AA8A594DF05D35759E07C9566EB14F2B224390A382D1D7049CC",
                                    m_l=(3, 4), v=2, L=2, Npsk=8, sbAmp=True)
    pmi = example_coef_factory_test(ant_dim=(4, 2), subbands='01111111111111111',
                                    x1="0xF7A0A",
                                    x2="0xAFAFAFAF83838383",
                                    m_l=(3, 0), v=1, L=2, Npsk=4, sbAmp=False)
    pmi = example_pmi_generator(subbands='01111111111111111', Npsk=8, sbAmp=True, v=2)
    pass