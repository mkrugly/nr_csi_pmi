from dataclasses import dataclass
import logging
import math
import numpy as np
import numpy.typing as npt
import pandas as pd
import random
from typing import Any

if __name__ == '__main__':
    from nr_csi_pmi.base import Base, BaseMatrix, PcsirsN1N2, I11, I12, Vm1m2
else:
    from .base import Base, BaseMatrix, PcsirsN1N2, I11, I12, Vm1m2


logger = logging.getLogger(__name__)


class ParamCombR16:
    """TS38.214 tab. Table 5.2.2.2.5-1

    """
    tab_v12: dict[int, tuple[int, float, float]] = {
           #L, pv_12, beta
        1: (2, 0.25, 0.25),
        2: (2, 0.25, 0.5 ),
        3: (4, 0.25, 0.25),
        4: (4, 0.25, 0.5 ),
        5: (4, 0.25, 0.75),
        6: (4, 0.5,  0.5 ),
        7: (6, 0.25, 0.5 ),
        8: (6, 0.25, 0.75),
    }

    tab_v34: dict[int, tuple[int, float, float]] = {
           #L, pv_34, beta
        1: (2, 0.125, 0.25),
        2: (2, 0.125, 0.5 ),
        3: (4, 0.125, 0.25),
        4: (4, 0.125, 0.5 ),
        5: (4, 0.25,  0.75),
        6: (4, 0.25,  0.5 ),
        7: (6, None,  0.5 ),
        8: (6, None,  0.75),
    }

    @classmethod
    def l_pv_beta(cls, comb_inx: int = 1, v: int = 1) -> tuple[int, float, float]:
        return cls.tab_v12.get(comb_inx, ()) if v in (1, 2) else cls.tab_v34.get(comb_inx, ())

    @classmethod
    def M_v(cls, comb_inx: int, R: int, N3: int, v: int = 1) -> int:
        pv = cls.l_pv_beta(comb_inx=comb_inx, v=v)[1]
        return math.ceil(pv*(N3/R))

    @classmethod
    def K0(cls, comb_inx: int, R: int, N3: int, v: int = 1) -> int:
        l, pv, beta = cls.l_pv_beta(comb_inx=comb_inx, v=v)
        M_1 = cls.M_v(comb_inx=comb_inx, R=R, N3=N3)
        return math.ceil(beta * 2 * l * M_1)


class K1P1:
    """
    38.214 Table 5.2.2.2.5-2
    """
    tab: list[float] = [
        0,  # actually reserved value
        1/(128**0.5),
        (1/8192)**0.25,
        0.125,
        (1/2048)**0.25,
        1/(2*(8**0.5)),
        (1/512)**0.25,
        0.25,
        (1/128)**0.25,
        1/(8**0.5),
        (1/32)**0.25,
        0.5,
        0.125**0.25,
        0.5**0.5,
        0.5**0.25,
        1
    ]

    @classmethod
    def p1(cls, k1: int) -> float:
        return cls.tab[k1] if 0 <= k1 < len(cls.tab) else 0


class K2P2:
    """
    38.214 Table 5.2.2.2.5-3
    """
    tab: list[float] = [
        1/(8*2**0.5),
        0.125,
        1/(4*2**0.5),
        0.25,
        1/(2*2**0.5),
        0.5,
        1/(2**0.5),
        1
    ]

    @classmethod
    def p2(cls, k2: int) -> float:
        return cls.tab[k2] if 0 <= k2 < len(cls.tab) else 0



class Yl(Base):
    # 38.214 Table 5.2.2.2.5-4
    C_table_r16 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 1, 0, 0, 0, 0, 0, 0],
        [4, 6, 4, 1, 0, 0, 0, 0, 0],
        [5, 10, 10, 5, 1, 0, 0, 0, 0],
        [6, 15, 20, 15, 6, 1, 0, 0, 0],
        [7, 21, 35, 35, 21, 7, 1, 0, 0],
        [8, 28, 56, 70, 56, 28, 8, 1, 0],
        [9, 36, 84, 126, 126, 84, 36, 9, 1],
        [10, 45, 120, 210, 252, 210, 120, 45, 10],
        [11, 55, 165, 330, 462, 462, 330, 165, 55],
        [12, 66, 220, 495, 792, 924, 792, 495, 220],
        [13, 78, 286, 715, 1287, 1716, 1716, 1287, 715],
        [14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002],
        [15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005],
        [16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440],
        [17, 136, 680, 2380, 6188, 12376, 19448, 24310, 24310],
        [18, 153, 816, 3060, 8568, 18564, 31824, 43758, 48620],
    ])

    def __init__(self, comb_inx: int, R: int, N3: int, v: int,
                 i16l_or_n3l: int|str|list[int], i15: int | str = 0):
        super().__init__()
        self._comb_inx: int = comb_inx
        self._R: int = R
        self._N3: int = N3
        self._v: int = v
        self._M_v: int = ParamCombR16.M_v(comb_inx=comb_inx, R=self._R, N3=self._N3, v=self._v)
        if self._N3 <= 19:
            self._i16l_size = self.bit_size_log(self._N3 - 1, self._M_v - 1)
            self._i15l_size = 0
        else:
            self._i16l_size = self.bit_size_log(2 * self._M_v - 1, self._M_v - 1)
            self._i15l_size = self.bit_size_log(2 * self._M_v)
        if isinstance(i15, str):
            self._i15: int = int(self.to_bin_str(i15, self._i15l_size), 2) if self._i15l_size else 0
        else:
            self._i15: int = i15

        if N3 > 19:
            self._M_init: int = self._i15 - 2*self._M_v if self._i15 else 0
        else:
            self._M_init: int = 0

        if isinstance(i16l_or_n3l, list):
            self.as_int = self.get_i16l(n3l=i16l_or_n3l, N3=self._N3, M_v=self._M_v, M_init=self._M_init)
            self._n3l: list[int] = i16l_or_n3l
        else:
            self.as_int = i16l_or_n3l
            self._n3l: list[int] = self.get_n3l(i16l=self.as_int, N3=self._N3, M_v=self._M_v, i15=self._i15)
        self._validate()
        self._matrix: npt.NDArray[np.complex128] = np.array([
            [np.exp((1j*2*np.pi*t*n3lf)/self._N3) for n3lf in self._n3l]
            for t in range(self._N3)])

    def _validate(self):
        pass

    def __str__(self):
        return f"{self.__class__}: i16l={self.i16l} ({self.i16l_bin}), i15={self.i15}, n3l:{self.n3l}\n{self.matrix}"

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        return self._matrix

    @property
    def shape(self) -> tuple[int,...]:
        return self.matrix.shape

    @property
    def size(self) -> int:
        return self._i16l_size

    @property
    def i16l(self) -> int:
        return self.as_int

    @property
    def i16l_bin(self) -> str:
        return self.as_bin

    @property
    def i15(self) -> int:
        return self._i15

    @property
    def i15_bin(self) -> str:
        return self.to_bin_str(self.i15, self._i15l_size)

    @property
    def n3l(self) -> list[int]:
        return self._n3l

    @classmethod
    def _check_m_init(cls, v: int, M_v):
        assert v <= 0 and abs(v) < 2*M_v, f"Wrong M_init ({v}) value (should be negative and < M_v:{M_v})"

    @classmethod
    def i16l_size(cls, comb_inx: int, R: int, N3: int, v: int) -> int:
        M_v = ParamCombR16.M_v(comb_inx=comb_inx, R=R, N3=N3, v=v)
        if N3 <= 19:
            return cls.bit_size_log(N3 - 1, M_v - 1)
        else:
            return cls.bit_size_log(2 * M_v - 1, M_v - 1)

    @classmethod
    def get_n3l(cls, i16l: int, N3: int, M_v: int, i15: int = 0) -> list[int]:
        s_prev = 0
        n3l: list[int] = [0]
        C = cls.c_tab(is_rel16=True)
        M_init = i15 - 2*M_v if N3 > 19 else 0
        cls._check_m_init(M_init, M_v)
        for i, f in enumerate(range(1, M_v)):
            sorter = np.argsort(C[:, M_v-1-f])
            x_star = int(np.searchsorted(C[:, M_v-1-f], i16l-s_prev, side="right", sorter=sorter) - 1)
            assert M_v-1-f <= x_star <= N3-1-f,  f"X_star ({x_star}) out of allowed range ({M_v-1-f}, {N3-1-f}). Check i16l value!"
            e_i = C[x_star, M_v-1-f]
            s_prev = s_prev + e_i
            if N3 <= 19:
                nl = N3-1-x_star
            else:
                nl = 2*M_v - 1 - x_star
                if nl > M_init + 2*M_v - 1:
                    nl += (N3 - 2*M_v)
            n3l.append(nl)
            logger.debug(f"find n3l, iter#{i}: x_star:{x_star}, e_i:{e_i}, n3l_{f}: {n3l}")
        return n3l

    @classmethod
    def get_i16l(cls, n3l: list[int], N3: int, M_v: int, M_init: int = 0) -> int:
        # first make sure to remove n3l==0 from the list, as it is not used in the following algorithm
        # because only the nonzero indices are reported and f is counted from 1
        _n3l = list(filter(lambda num: num != 0, sorted(n3l)))
        cls._check_m_init(M_init, M_v)
        C = cls.c_tab(is_rel16=True)
        if M_v == 1:
            i16l = 0
        else:
            if N3 <= 19:
                i16l = sum([C[N3 - 1 - _n3l[i], M_v-1-f] for i, f in enumerate(range(1, M_v))])
            else:
                ints = {(M_init + i) % N3 for i in range(2*M_v)}
                i16l = 0
                for i, f in enumerate(range(1, M_v)):
                    if _n3l[i] not in ints:
                        continue
                    if _n3l[i] <= M_init + 2*M_v - 1:
                        i16l += C[2*M_v - 1 - _n3l[i], M_v-1-f]
                    elif _n3l[i] > M_init + N3 - 1:
                        i16l += C[N3 - 1 - _n3l[i], M_v - 1 - f]
        return int(i16l)

    @classmethod
    def c_tab(cls, is_rel16: bool = False):
        return cls.C_table_r16

    @classmethod
    def n3l_generator(cls, comb_inx: int, R: int, N3: int, M_init: int | None = None,
                      v: int = 1, randomize: bool = False) -> dict[int, dict[str, int | list[int]]]:
        M_v = ParamCombR16.M_v(comb_inx=comb_inx, R=R, N3=N3, v=v)
        n3_values = []
        n3_set = []
        n3_dict: dict[int, dict[str, int | list[int]]] = {}

        def add(i15, n3l_values, n3l, m_init=None):
            n3_dict[i15] = {"i15": i15, "n3l_values": n3l_values, "n3l": n3l, "M_init": m_init}

        if N3 <= 19:
            i15 = 0
            n3_values.append(list(range(N3)))
            _selection = [0] + random.sample(n3_values[-1][1:], M_v-1) if randomize else n3_values[-1][:M_v]
            n3_set.append(sorted(_selection))
            add(i15, n3_values[-1], n3_set[-1])
        else:
            M_init = [M_init] if M_init is not None else [-1 * i for i in reversed(range(2*M_v))]
            for M_initial in M_init:
                i15 = M_initial + 2*M_v if M_initial else 0
                n3_values.append(sorted([(M_initial+i) % N3 for i in range(2*M_v)]))
                _selection = [0] + random.sample([i for i in n3_values[-1] if i != 0], M_v-1) if randomize \
                    else n3_values[-1][:M_v]
                n3_set.append(sorted(_selection))
                add(i15, n3_values[-1], n3_set[-1], m_init=M_initial)
        return n3_dict

    @classmethod
    def factory(cls, comb_inx: int, R: int, N3: int, v: int,
                i16: str | list[str|list[int]],
                i15: int | str = 0):
        if isinstance(i16, str):
            _size_l = cls.i16l_size(comb_inx=comb_inx, R=R, N3=N3, v=v)
            _i16 = cls.to_bin_str(i16, _size_l * v)
            i16 = [_i16[i:i + _size_l] for i in range(0, len(_i16), _size_l)]
        return [cls(comb_inx=comb_inx, R=R, N3=N3, v=v, i16l_or_n3l=i16l, i15=i15) for i16l in i16]


class Y:
    def __init__(self, yl: list[Yl]):
        self._yl: list[Yl] = yl

    def __str__(self):
        return f"{self.__class__}:i16l={self.i16l_list} ({self.i16l_bin_list}), n3l:{self.n3l_list}"

    def _i16_for_l(self, l: int) -> int | None:
        _lst = self.i16l_list
        return _lst[l-1] if 0 < l <= len(_lst) else None

    @property
    def size(self) -> int:
        return len(self.as_list)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.get_yl_for_l(l=1).shape

    @property
    def as_list(self) -> list[Yl]:
        return self._yl

    @property
    def i15(self) -> int | None:
        return self._yl[-1].i15 if self.size else None

    @property
    def i15_bin(self) -> int | None:
        return self._yl[-1].i15_bin if self.size else None

    @property
    def i16(self) -> str:
        return "".join([yl.as_bin for yl in self.as_list])

    @property
    def i16l_list(self) -> list[int]:
        return [yl.i16l for yl in self.as_list]

    @property
    def i16l_bin_list(self) -> list[str]:
        return [yl.i16l_bin for yl in self.as_list]

    @property
    def i161(self) -> int | None:
        return self._i16_for_l(1)

    @property
    def i162(self) -> int | None:
        return self._i16_for_l(2)

    @property
    def i163(self) -> int | None:
        return self._i16_for_l(3)

    @property
    def i164(self) -> int | None:
        return self._i16_for_l(4)

    @property
    def n3l_list(self) -> list[list[int]]:
        return [yl.n3l for yl in self.as_list]

    @property
    def yl_matrices(self) -> list[npt.NDArray]:
        return [yl.matrix for yl in self.as_list]

    def get_yl_for_l(self, l: int) -> Yl:
        assert 0 < l <= len(self.as_list), f"Wrong layer l:{l} (range: 0 .. {len(self.as_list)})"
        return self.as_list[l - 1]


class I23l(Base):
    def __init__(self, v: int, i23: str | list[int | str]):
        super().__init__()
        self._v = v
        self._size_l = 4
        self._size = self._size_l * v
        self._i23l: str = ""
        self._i23l_list: list[int] = []
        self._from_i23(i23)
        self._validate()
        self._p1_list: list[float] = [K1P1.p1(k1) for k1 in self._i23l_list]

    def __str__(self):
        return f"{self.__class__}: {self.as_bin}, k1={self.k1l}, p1={self.p1l}"

    def _from_i23(self, i23: str | list[int | str]):
        if isinstance(i23, list):
            i23 = "".join([self.to_bin_str(v=i, size=self._size_l) for i in i23])
        self.as_bin = i23
        self._i23l_list = [int(self.as_bin[i * self._size_l: (i + 1) * self._size_l], 2) for i in range(self._v)]

    def _validate(self):
        assert len(self.as_bin) == self._size, f"i23 bit string length ({len(self.as_bin)}) out of range ({self._size})"
        _v_max = 2 ** self._size_l
        for i, v in enumerate(self._i23l_list):
            assert 0 <= v < _v_max, f"i23{i + 1}:{v} value out of range (min: 0, max:{_v_max - 1})"

    @property
    def i231(self) -> int | None:
        return self.i23l_for_l(1)

    @property
    def i232(self) -> int | None:
        return self.i23l_for_l(2)

    @property
    def i233(self) -> int | None:
        return self.i23l_for_l(3)

    @property
    def i234(self) -> int | None:
        return self.i23l_for_l(4)

    @property
    def k1l(self) -> list[int]:
        return self._i23l_list

    @property
    def p1l(self) -> list[float]:
        return self._p1_list

    def i23l_for_l(self, l: int) -> int | None:
        if len(self._i23l_list) < l or l < 1:
            return None
        return self._i23l_list[l - 1]

    def p1_for_l(self, l: int) -> float | None:
        if len(self._i23l_list) < l or l < 1:
            return None
        return K1P1.p1(self._i23l_list[l - 1])


class I24l(Base):
    def __init__(self, v: int, K_NZ: int, i24: str | list[int]):
        super().__init__()
        self._v = v
        self._K_NZ = K_NZ
        self._num_of_items = K_NZ - v
        self._size_rep = 3
        self._size = self._size_rep * self._num_of_items
        self._i24l_list: list[int] = []
        self._from(i24)
        self._validate()
        self._p2_list: list[float] = [K2P2.p2(k2) for k2 in self._i24l_list]

    def __str__(self):
        return f"{self.__class__}: {self.as_bin}, k2={self.k2l}, p2={self.p2l}"

    def _from(self, i24: str | list[int]):
        if isinstance(i24, list):
            self._i24l_list = i24
            self.as_bin = "".join([self.to_bin_str(i, self._size_rep) for i in i24])
        elif isinstance(i24, str):
            self.as_bin = i24
            self._i24l_list = [int(self.as_bin[i * self._size_rep: (i + 1) * self._size_rep], 2)
                               for i in range(self._num_of_items)]

    def _validate(self):
        assert len(self.as_bin) == self._size, f"i24l bit string length ({len(self.as_bin)}) out of range ({self._size})"
        _v_max = 2 ** self._size_rep
        for i, v in enumerate(self._i24l_list):
            assert 0 <= v < _v_max, f"i24l:{v} {i}th value out of range (min:0, max:{_v_max - 1})"

    @property
    def k2l(self) -> list[int]:
        return self._i24l_list

    @property
    def p2l(self) -> list[float]:
        return self._p2_list

    def from_pos(self, pos: npt.NDArray | list[int], as_bin_str: bool = False) -> str | None:
        try:
            a = np.array(self.k2l)[pos]
            shifter = np.array([i * self._size_rep for i in reversed(range(0, len(a)))])
            _size = len(a) * self._size_rep
            if as_bin_str:
                return self.to_bin_str(int(np.sum(a << shifter)), _size)
            else:
                return hex(self.align_bit_length(int(np.sum(a << shifter)), _size))
        except Exception as e:
            return None


class I25l(Base):
    def __init__(self, v: int, K_NZ: int, i25: str | list[int]):
        super().__init__()
        self._v = v
        self._K_NZ = K_NZ
        self._num_of_items = K_NZ - v
        self._size_rep = 4
        self._size = self._size_rep * self._num_of_items
        self._i25l_list: list[int] = []
        self._from(i25)
        self._validate()
        self._phi_list: list[float] = [np.around(np.exp((1j * 2 * np.pi * c) / 16), 3) for c in self._i25l_list]

    def __str__(self):
        return f"{self.__class__}: {self.as_bin}, c={self.C_phil}, phi={self.phil}"

    def _from(self, i25: str | list[int]):
        if isinstance(i25, list):
            self._i25l_list = i25
            self.as_bin = "".join([self.to_bin_str(i, self._size_rep) for i in i25])
        elif isinstance(i25, str):
            self.as_bin = i25
            self._i25l_list = [int(self.as_bin[i * self._size_rep: (i + 1) * self._size_rep], 2)
                               for i in range(self._num_of_items)]

    def _validate(self):
        assert len(self.as_bin) == self._size, f"i25l bit string length ({len(self.as_bin)}) out of range ({self._size})"
        _v_max = 2 ** self._size_rep
        for i, v in enumerate(self._i25l_list):
            assert 0 <= v < _v_max, f"i25l:{v} {i}th value out of range (min:0, max:{_v_max - 1})"

    @property
    def C_phil(self) -> list[int]:
        return self._i25l_list

    @property
    def phil(self) -> list[float]:
        return self._phi_list

    def from_pos(self, pos: npt.NDArray | list[int], as_bin_str: bool = False) -> str | None:
        try:
            a = np.array(self.C_phil)[pos]
            shifter = np.array([i * self._size_rep for i in reversed(range(0, len(a)))])
            _size = len(a) * self._size_rep
            if as_bin_str:
                return self.to_bin_str(int(np.sum(a << shifter)), _size)
            else:
                return hex(self.align_bit_length(int(np.sum(a << shifter)), _size))
        except Exception as e:
            return None


class I18l(Base):
    def __init__(self, L: int, v: int, K_NZ: int, N3: int, i18: str | list[int | str]):
        super().__init__()
        self._L = L
        self._v = v
        self._N3 = N3
        self._K_NZ = K_NZ
        self._size_l = self.bit_size_log(K_NZ if v == 1 else 2 * L)
        self._size = self._size_l * v
        self._i18l_list: list[int] = []
        self._i18l_list_flat: list[int] = []
        self._i18l_lifs: list[tuple[int,...]] = []
        self._from_i18(i18)
        self._validate()

    def _from_i18(self, i18: str | list[int | str]):
        if isinstance(i18, list):
            i18 = "".join([self.to_bin_str(v=i, size=self._size_l) for i in i18])
        self.as_bin = i18
        for i in range(self._v):
            l = i + 1
            v = int(self.as_bin[i * self._size_l: l * self._size_l], 2)
            v_flat = v + i * 2 ** self._size_l
            self._i18l_list.append(v)
            self._i18l_list_flat.append(v_flat)
            self._i18l_lifs.append((l, v, 0))

    def _validate(self):
        assert len(self.as_bin) == self._size, f"i18l bit string length ({len(self.as_bin)}) != {self._size}"
        _v_max = 2 ** self._size_l
        for i, v in enumerate(self._i18l_list):
            assert v < _v_max, f"i18{i + 1}:{v} value out of range (max:{_v_max})"

    def __str__(self):
        return f"{self.__class__}: {self.as_bin} ({self._i18l_list}), lif: {self.i18l_lif},  flat: {self.i18l_flat}"

    @property
    def i181(self) -> int:
        return self.i18l_for_l(1)

    @property
    def i182(self) -> int:
        return self.i18l_for_l(2)

    @property
    def i183(self) -> int:
        return self.i18l_for_l(3)

    @property
    def i184(self) -> int:
        return self.i18l_for_l(4)

    @property
    def i18l(self) -> list[int]:
        return self._i18l_list

    @property
    def i18l_lif(self) -> list[tuple[Any]]:
        return self._i18l_lifs

    @property
    def i18l_flat(self) -> list[int]:
        return self._i18l_list_flat

    def i18l_for_l(self, l: int, flat_inx=False) -> int | None:
        if len(self._i18l_list) < l or l < 1:
            return None
        return self._i18l_list_flat[l - 1] if flat_inx else self._i18l_list[l - 1]


class I17l(Base):
    def __init__(self, n3l: list[list[int]], L: int, N3: int, v: int, K0: int,
                 i17_prio: str = None, i17_flat: str | list[str] = None,
                 i17_lifs: list[tuple[int]] = None):
        super().__init__()
        self._L = L
        self._N3: int = N3
        self._v: int = v
        self._n3l: list[list[int]] = n3l
        self._M_v: int = len(n3l[0])
        self._size_l: int = 2 * L * self._M_v
        self._size: int = self._size_l * v
        self.K0: int = K0
        self._lif_inx_map: pd.DataFrame = self.lif_to_inx_map(n3l=n3l, L=L, N3=N3, v=v)
        self._i17_prio_arr: npt.NDArray[int] = np.zeros(self._size, dtype=int)
        self._i17_flat_arr: npt.NDArray[int] = np.zeros(self._size, dtype=int)
        self._i17_prio_pos: npt.NDArray[int] = np.array([], dtype=int)
        self._i17_flat_pos: npt.NDArray[int] = np.array([], dtype=int)
        self._i17_lif: pd.DataFrame = None
        if i17_prio:
            self._from_prio(i17=i17_prio)
        elif i17_flat:
            self._from_flat(i17=i17_flat)
        elif i17_lifs:
            self._from_lif(i17=i17_lifs)
        self._validate()

    def _validate(self):
        assert self.K_NZ <= 2 * self.K0, f"K_NZ ({self.K_NZ}) exceeds 2*K0 ({2 * self.K0})"
        assert self.K_NZ_1 <= self.K0, f"K_NZ_1 ({self.K_NZ_1}) exceeds K0 ({self.K0})"
        assert self.K_NZ_2 <= self.K0, f"K_NZ_2 ({self.K_NZ_2}) exceeds K0 ({self.K0})"
        assert self.K_NZ_3 <= self.K0, f"K_NZ_3 ({self.K_NZ_3}) exceeds K0 ({self.K0})"
        assert self.K_NZ_4 <= self.K0, f"K_NZ_1 ({self.K_NZ_4}) exceeds K0 ({self.K0})"

    def _from_prio(self, i17: str):
        # align size first
        _i17 = self.to_bin_str(v=i17, size=self._size)
        self._i17_prio_arr = np.array(list(map(int, list(_i17))), dtype=int)
        # find positions with 1's
        self._i17_prio_pos: npt.NDArray[int] = np.where(self._i17_prio_arr)[0]
        # locate lif_inx map entries based on the index
        self._i17_lif: pd.DataFrame = self._lif_inx_map.query("prio_inx in @self._i17_prio_pos")
        # update flat positions
        self._i17_flat_pos: npt.NDArray[int] = np.array(self._i17_lif.get("flat_inx"), dtype=int)
        # update flat array
        self._i17_flat_arr[self._i17_flat_pos] = 1

    def _from_flat(self, i17: str | list[str]):
        if isinstance(i17, list):
            i17 = "".join([self.to_bin_str(v=i, size=self._size_l) for i in i17])
        # align size first
        _i17 = self.to_bin_str(v=i17, size=self._size)
        self._i17_flat_arr = np.array(list(map(int, list(_i17))), dtype=int)
        # find positions with 1's
        self._i17_flat_pos: npt.NDArray[int] = np.where(self._i17_flat_arr)[0]
        # locate lif_inx map entries based on the index
        self._i17_lif: pd.DataFrame = self._lif_inx_map.query("flat_inx in @self._i17_flat_pos")
        # update prio positions
        self._i17_prio_pos: npt.NDArray[int] = np.array(self._i17_lif.get("prio_inx"), dtype=int)
        # update prio array
        self._i17_prio_arr[self._i17_prio_pos] = 1

    def _from_lif(self, i17: list[tuple[int]]):
        _prio_pos = []
        _flat_pos = []
        for lif in i17:
            _flat_pos.append(self.pos_for_lif(lif=lif, is_prio=False))
            _prio_pos.append(self.pos_for_lif(lif=lif, is_prio=True))
        self._i17_prio_pos = np.array(_prio_pos, dtype=int)
        self._i17_flat_pos = np.array(_flat_pos, dtype=int)
        self._i17_prio_arr[self._i17_prio_pos] = 1
        self._i17_flat_arr[self._i17_flat_pos] = 1
        self._i17_lif: pd.DataFrame = self._lif_inx_map.query("flat_inx in @self._i17_flat_pos")

    def __str__(self):
        return f"{self.__class__}: {self.as_bin}\ni17_flat: {self.bin_flat}\ni17 lifs:\n{self.lifs}"

    @property
    def v(self) -> int:
        return self._v

    @property
    def L(self) -> int:
        return self._L

    @property
    def M_v(self) -> int:
        return self._M_v

    @property
    def as_bin(self) -> str:
        return "".join(list(map(str, self._i17_prio_arr)))

    @property
    def as_hex(self) -> str:
        return hex(self.as_int)

    @property
    def as_int(self) -> int:
        return int(self.as_bin, 2)

    @property
    def bin_flat(self) -> str:
        return "".join(list(map(str, self._i17_flat_arr)))

    @property
    def i171_bin(self) -> str:
        return self.i17l_for_l(1)

    @property
    def i172_bin(self) -> str:
        return self.i17l_for_l(2)

    @property
    def i173_bin(self) -> str:
        return self.i17l_for_l(3)

    @property
    def i174_bin(self) -> str:
        return self.i17l_for_l(4)

    @property
    def i171(self) -> str:
        return self.i17l_for_l(1, True)

    @property
    def i172(self) -> str:
        return self.i17l_for_l(2, True)

    @property
    def i173(self) -> str:
        return self.i17l_for_l(3, True)

    @property
    def i174(self) -> str:
        return self.i17l_for_l(4, True)

    @property
    def K_NZ(self) -> int:
        return self.flat_pos.size

    @property
    def K_NZ_1(self) -> int:
        return len(self.lifs_for_l(1))

    @property
    def K_NZ_2(self) -> int:
        return len(self.lifs_for_l(2))

    @property
    def K_NZ_3(self) -> int:
        return len(self.lifs_for_l(3))

    @property
    def K_NZ_4(self) -> int:
        return len(self.lifs_for_l(4))

    @property
    def flat_pos(self) -> npt.NDArray[int]:
        return self._i17_flat_pos

    @property
    def prio_pos(self) -> npt.NDArray[int]:
        return self._i17_prio_pos

    @property
    def lifs(self) -> pd.DataFrame:
        return self._i17_lif

    def i17l_for_l(self, l: int, as_hex: bool = False) -> str:
        b = self.bin_flat
        start = self._size_l * (l - 1)
        end = self._size_l * l
        s = b[start:end] if len(b) >= end else ""
        return hex(int(s, 2)) if s and as_hex else s

    def lifs_for_l(self, l: int) -> pd.DataFrame:
        return self.lifs[self.lifs.index.get_level_values(0) == l]

    def lif_for_pos(self, pos: int|list[int], is_prio=False, only_inx=True) -> pd.DataFrame | list[tuple[Any]]:
        col = "prio_inx" if is_prio else "flat_inx"
        if isinstance(pos, (int, np.integer)):
            pos = [pos]
        _v = self._lif_inx_map.loc[self._lif_inx_map[col].isin(pos)]
        if only_inx:
            _v = _v.index.values
            return list(_v) if _v.size else list(tuple())
        else:
            return _v

    def lif_for_inx(self, inx: tuple[Any] | list[tuple[Any]]) -> pd.DataFrame:
        if isinstance(inx, tuple):
            inx = [inx]
        return self._lif_inx_map.loc[self._lif_inx_map.index.isin(inx)]

    def pos_for_lif(self, lif: tuple[Any], is_prio=False) -> int:
        col = "prio_inx" if is_prio else "flat_inx"
        return self._lif_inx_map[col].get(lif)

    @classmethod
    def flat_inx(cls, l: int, i: int, f: int, L: int, M_v: int):
        return (l - 1) * 2 * L * M_v + 2 * L * f + i

    @classmethod
    def pri_inx(cls, l: int, i: int, f: int, n3l: list[list[int]], L: int, N3: int, v: int):
        return cls.pri_for_all(n3l=n3l, L=L, N3=N3, v=v).get((l, i, f), None)

    @classmethod
    def pri(cls, l: int, i: int, f: int, n3l: list[int], L: int, N3: int, v: int) -> int:
        _n3l_f = n3l[f]
        _pi_f = min(2 * _n3l_f, 2 * (N3 - _n3l_f) - 1)
        return 2 * L * v * _pi_f + v * i + l

    @classmethod
    def pri_for_all(cls, n3l: list[list[int]], L: int, N3: int, v: int) -> dict[tuple[int, int, int], tuple[int, int, int]]:
        d = {}
        M_v = len(n3l[0])
        for l in range(1, v + 1):
            for i in range(2 * L):
                for f in range(M_v):
                    _pri = cls.pri(l=l, i=i, f=f, n3l=n3l[l - 1], L=L, N3=N3, v=v)
                    d[(l, i, f)] = _pri
        to_ret = {v[0]: (i, cls.flat_inx(l=v[0][0], i=v[0][1], f=v[0][-1], L=L, M_v=M_v), v[1]) for i, v in
                  enumerate(sorted(d.items(), key=lambda item: item[1]))}
        return to_ret

    @classmethod
    def lif_to_inx_map(cls, n3l: list[list[int]], L: int, N3: int, v: int) -> pd.DataFrame:
        _in = cls.pri_for_all(n3l=n3l, L=L, N3=N3, v=v)
        inx = pd.MultiIndex.from_tuples(_in.keys(), names=["l", "i", "f"])
        return pd.DataFrame(_in.values(), index=inx, columns=["prio_inx", "flat_inx", "prio"])

    @classmethod
    def factory(cls, n3l: list[list[int]], L: int, N3: int, v: int, K0: int,
                i17: str | list[str] | list[tuple[int]]):
        if i17:
            if isinstance(i17, list):
                if isinstance(i17[0], str):
                    return cls(i17_flat=i17, n3l=n3l, L=L, N3=N3, v=v, K0=K0)
                else:
                    return cls(i17_lifs=i17, n3l=n3l, L=L, N3=N3, v=v, K0=K0)
            else:
                return cls(i17_prio=i17, n3l=n3l, L=L, N3=N3, v=v, K0=K0)
        return None


class P1(BaseMatrix):
    def __init__(self, i23: I23l, i17: I17l, i18: I18l):
        self.i17l: I17l = i17
        self.i18l: I18l = i18
        self.i23l: I23l = i23
        # matrix of shape (v, 2L) to be able to use flat indices from i17 to address the necessary fields
        super().__init__(shape=(self.i17l.v, 2 * self.i17l.L), dtype=np.float64, vsplit=self.i17l.v,
                         shape_desc="v, 2L")
        self._init()

    def _init(self):
        for l, i, f in self.strongest.index.values:
            L = self.i17l.L
            p1 = self.i23l.p1_for_l(l)
            self.matrix[l - 1] = [1] * L + [p1] * L if i < L else [p1] * L + [1] * L

    def __str__(self):
        return (f"{self.__class__}: strongest:\n{self.strongest}\nstrongest_xpol:\n{self.strongest_xpol}\n"
                f"matrix {self.shape}:\n{self.matrix}")

    @property
    def strongest(self) -> pd.DataFrame:
        return self.i17l.lif_for_inx(inx=self.i18l.i18l_lif)

    @property
    def strongest_xpol(self) -> pd.DataFrame:
        _func = lambda x: x + self.i17l.L if x % (2 * self.i17l.L) < self.i17l.L else x - self.i17l.L
        _flat = list(map(_func, self.strongest.get("flat_inx").values))
        return self.i17l.lif_for_pos(pos=_flat, only_inx=False)

    @property
    def k1l(self) -> list[int]:
        return self.i23l.k1l

    def for_l(self, l: int) -> npt.NDArray:
        return self.matrix[l - 1]


class BaseSb(BaseMatrix):
    def __init__(self, i17: I17l, i18: I18l, dtype: npt.DTypeLike, sb_coef):
        self.i17l: I17l = i17
        self.i18l: I18l = i18
        self._sb_coef = sb_coef
        # matrix of shape (v*M_v, 2L) to be able to use flat indices from i17 to address the necessary fields
        super().__init__((self.i17l.v * self.i17l.M_v, 2 * self.i17l.L), dtype=dtype,
                         vsplit=self.i17l.v, shape_desc="v*M_v, 2L")

    def _init_df_inx(self):
        lev0 = np.repeat(range(1, self.i17l.v + 1), self.i17l.M_v)
        lev1 = np.tile(range(self.i17l.M_v), self.i17l.v)
        pd.MultiIndex.from_arrays([lev0, lev1])

    def __str__(self):
        return f"{self.__class__}: lifs:\n{self.other}\nmatrix {self.shape} ({self.shape_desc}):\n{self.matrix}"

    @property
    def strongest(self) -> pd.DataFrame:
        return self.i17l.lif_for_inx(inx=self.i18l.i18l_lif)

    @property
    def strongest_xpol(self) -> pd.DataFrame:
        _func = lambda x: x + self.i17l.L if x % (2 * self.i17l.L) < self.i17l.L else x - self.i17l.L
        _flat = list(map(_func, self.strongest.get("flat_inx").values))
        return self.i17l.lif_for_pos(pos=_flat, only_inx=False)

    @property
    def other(self) -> pd.DataFrame:
        _strongest_lifs = self.strongest.index.values
        return self.i17l.lifs.loc[~self.i17l.lifs.index.isin(_strongest_lifs)]

    def f_x_i_for_l(self, l: int) -> npt.NDArray:
        assert 0 < l <= self.i17l.v
        return self.matrices[l - 1]

    def for_l(self, l: int) -> npt.NDArray:
        return self.f_x_i_for_l(l).T

    def _coef_for_l(self, l: int, as_bin_str: bool = False) -> str | None:
        try:
            return self._sb_coef.from_pos(pos=self.other.index.get_loc(l), as_bin_str=as_bin_str)
        except KeyError as e:
            return None

    def _flat(self) -> list[str]:
        return [self._coef_for_l(i) for i in range(1, 5)]

    def _flat_bin(self, as_str: bool = True) -> str | list[str]:
        _to_ret = [self._coef_for_l(i, True) for i in range(1, 5)]
        if as_str:
            _to_ret = "".join([i for i in _to_ret if i is not None])
        return _to_ret


class P2(BaseSb):
    def __init__(self, i17: I17l, i18: I18l, i24: I24l):
        super().__init__(i17=i17, i18=i18, dtype=np.float64, sb_coef=i24)
        self._init()

    def _init(self):
        _strongest = self.strongest.get("flat_inx").to_numpy()
        np.put(self._matrix, _strongest, [1] * len(_strongest))
        # MK this is not needed for the SB matrix
        # _strongest_xpol = self.strongest_xpol.get("flat_inx").to_numpy()
        # np.put(self._matrix, _strongest_xpol, [1] * len(_strongest))
        _other = self.other.get("flat_inx").to_numpy()
        np.put(self._matrix, _other, self.i24l.p2l)

    @property
    def i24l(self) -> I24l:
        return self._sb_coef

    @property
    def i24l_flat_bin(self) -> str:
        return self._flat_bin()

    @property
    def i24l_flat_bin_list(self) -> list[str]:
        return self._flat_bin(as_str=False)

    @property
    def i24l_flat(self) -> list[str]:
        return self._flat()

    @property
    def i241(self) -> str:
        return self._flat()[0]

    @property
    def i242(self) -> str:
        return self._flat()[1]

    @property
    def i243(self) -> str:
        return self._flat()[2]

    @property
    def i244(self) -> str:
        return self._flat()[3]

    @property
    def k2l(self) -> list[int]:
        return self.i24l.k2l


class Phi(BaseSb):
    def __init__(self, i17: I17l, i18: I18l, i25: I25l):
        super().__init__(i17=i17, i18=i18, dtype=np.complex128, sb_coef=i25)
        self._init()

    def _init(self):
        _strongest = self.strongest.get("flat_inx").to_numpy()
        np.put(self._matrix, _strongest, [1] * len(_strongest))
        # MK this is not needed for the SB matrix
        # _strongest_xpol = self.strongest_xpol.get("flat_inx").to_numpy()
        # np.put(self._matrix, _strongest_xpol, [1] * len(_strongest))
        _other = self.other.get("flat_inx").to_numpy()
        _phi_other = [np.around(np.exp((1j * 2 * np.pi * c) / 16), 3) for c in self.i25l.C_phil]
        np.put(self._matrix, _other, _phi_other)

    @property
    def i25l(self) -> I25l:
        return self._sb_coef

    @property
    def i25l_flat_bin(self) -> str:
        return self._flat_bin()

    @property
    def i25l_flat_bin_list(self) -> list[str]:
        return self._flat_bin(as_str=False)

    @property
    def i25l_flat(self) -> list[str]:
        return self._flat()

    @property
    def i251(self) -> str:
        return self._flat()[0]

    @property
    def i252(self) -> str:
        return self._flat()[1]

    @property
    def i253(self) -> str:
        return self._flat()[2]

    @property
    def i254(self) -> str:
        return self._flat()[3]

    @property
    def C_phil(self) -> list[int]:
        return self.i25l.C_phil


class YP2Phitl(BaseMatrix):
    def __init__(self, ytl: Y, p2: P2, phi: Phi):
        self.ytl: Y = ytl
        self.p2: P2 = p2
        self.phi = phi
        # matrix of shape N3*v, 2L
        _matrix_shape = (self.ytl.size * self.ytl.shape[0], p2.shape[1])
        super().__init__(shape=_matrix_shape, dtype=np.complex128, vsplit=self.ytl.size, shape_desc="v*N3, 2L")
        self._init()

    def _init(self):
        out = []
        for i, yt in enumerate(self.ytl.as_list):
            prod = np.matmul(yt.matrix, np.multiply(self.p2.matrices[i], self.phi.matrices[i]))
            out.append(prod)
        self._matrix = np.vstack(out)

    def for_l(self, l: int) -> npt.NDArray:
        assert 0 < l <= self.ytl.size
        return self.matrices[l - 1]


class P1Yp2Phitl(BaseMatrix):
    def __init__(self, yp2phitl: YP2Phitl, p1: P1):
        self.yp2phitl: YP2Phitl = yp2phitl
        self.p1: P1 = p1
        # matrix of shape (N3*v, 2L)
        super().__init__(shape=self.yp2phitl.shape, dtype=self.yp2phitl.dtype,
                         vsplit=self.p1.shape[0], shape_desc="v*N3, 2L")
        self._gamma: npt.NDArray = np.zeros(self.shape[0], dtype=np.float64)
        self._init()

    def _init(self):
        # each row of the p1 matrix (representing p1 values for a certain layer)
        # shall be repeated N3 number of times (i.e. to be applied for all subbands)
        num_rep = int(self.shape[0] / self.p1.shape[0])
        p1_repeated = np.repeat(self.p1.matrix, num_rep, axis=0)
        self._matrix = np.multiply(p1_repeated, self.yp2phitl.matrix)
        _matrix_2 = np.square(np.multiply(p1_repeated, self.yp2phitl.matrix_abs))
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
    def __init__(self, vm1m2: Vm1m2, p1yp2phitl: P1Yp2Phitl):
        self.vm1m2: Vm1m2 = vm1m2
        self.p1yp2phitl: P1Yp2Phitl = p1yp2phitl
        self._N3: int = int(self.p1yp2phitl.shape[0] / self.p1yp2phitl.vsplit)
        self._v: int = self.p1yp2phitl.vsplit
        # matrix of shape (2*N1*N2, N3*v)
        super().__init__(shape=(self.vm1m2.vlm_w1.shape[0], self.p1yp2phitl.shape[0]), dtype=self.p1yp2phitl.dtype,
                         shape_desc="2*N1*N2, N3*v")
        self._init()

    def _init(self):
        # Vm1m2 is common for all layers and has shape 2N1N2 x 2L
        # P1Yp2Phitl includes coefficients for all subbands and all layers and has shape v*N3 x 2L
        # therefore for the matrix multiplication, the P1Yp2Phitl has to be transposed
        # (this way we will get the precoding vectors for all subbands and layers in columns
        _factor = (1 / np.sqrt(self.N1 * self.N2 * self.p1yp2phitl.gamma_matrix))
        self._matrix = _factor * np.matmul(self.vm1m2.vlm_w1, self.p1yp2phitl.matrix_T)
        # additional scaling for v > 1 - see 38.214 Table 5.2.2.2.5-5
        self._matrix *= self._get_w_scaling()

    def _get_w_scaling(self) -> float:
        _lst = (1, 1 / (2 ** 0.5), 1 / (3 ** 0.5), 0.5)
        return _lst[self._v - 1] if 0 < self._v <= len(_lst) else 1

    @property
    def N3(self) -> int:
        return self._N3

    @property
    def N1(self) -> int:
        return self.vm1m2.i12.N1

    @property
    def N2(self) -> int:
        return self.vm1m2.i12.N2

    def for_sb(self, inx: int) -> npt.NDArray:
        assert 0 <= inx < self.N3, f"Wrong subband (values 0..{self.N3 - 1} allowed)"
        return self.matrix[:, inx::self.N3]

    def for_layer(self, l: int) -> npt.NDArray:
        assert 0 < l <= self.p1yp2phitl.vsplit, f"Wrong layer (values 1..{self.p1yp2phitl.vsplit} allowed)"
        start = (l - 1) * self.N3
        end = start + self.N3
        return self.matrix[:, start:end]


@dataclass
class Pmi:
    w: W
    i11: I11
    i12: I12
    i17: I17l
    i18: I18l
    i23: I23l
    i24: I24l
    i25: I25l

    @property
    def vm1m2(self) -> Vm1m2:
        return self.w.vm1m2

    @property
    def p1yp2phitl(self) -> P1Yp2Phitl:
        return self.w.p1yp2phitl

    @property
    def p1(self) -> P1:
        return self.p1yp2phitl.p1

    @property
    def yp2phitl(self) -> YP2Phitl:
        return self.p1yp2phitl.yp2phitl

    @property
    def ytl(self) -> Y:
        return self.yp2phitl.ytl

    @property
    def p2(self) -> P2:
        return self.yp2phitl.p2

    @property
    def phi(self) -> Phi:
        return self.yp2phitl.phi

    @property
    def coefs(self) -> dict[str, Any]:
        return {
            "i11": self.i11.as_int,
            "i12": self.i12.as_int,
            "i23": self.i23.k1l,
            "i18": self.i18.i18l,
            "i15": self.ytl.i15,
            "i16": self.ytl.i16l_list,
            "i24": self.p2.i24l_flat,
            "i25": self.phi.i25l_flat,
            "i17": [self.i17.i171, self.i17.i172, self.i17.i173, self.i17.i174],
        }

    @property
    def coefs_bin(self) -> dict[str, Any]:
        return {
            "i11": self.i11.as_bin,
            "i12": self.i12.as_bin,
            "i23": self.i23.as_bin,
            "i18": self.i18.as_bin,
            "i15": self.ytl.i15_bin,
            "i16": self.ytl.i16,
            "i24": self.i24.as_bin,
            "i25": self.i25.as_bin,
            "i17": self.i17.as_bin,
        }

    @property
    def coefs_str(self) -> str:
        """
        "[i11 0100, i12 0001000, i18 001010011100, i23 0001110001010111, i15 0, "
               "i16 100101001010100101001010100101001010100101001010, "
               "i24 011010010100, i25 1111111110111000, i17 C000000000A00000000090000000008800000000]"
        """
        i17_str = "".join([i.replace("0x", "")
                           for i in [self.i17.i171, self.i17.i172, self.i17.i173, self.i17.i174]]).upper()
        return (f"[i11 {self.i11.as_bin}, i12 {self.i12.as_bin}, i18 {self.i18.as_bin}, i23 {self.i23.as_bin}, "
                f"i15 {self.ytl.i15_bin}, "
                f"i16 {self.ytl.i16}, i24 {self.p2.i24l_flat_bin}, i25 {self.phi.i25l_flat_bin}, i17 {i17_str}]")

    @property
    def beams(self) -> dict[str, Any]:
        return {
            "q1q2": self.i11.q1q2,
            "n1n2": self.i12.n1n2,
            "n3l": self.ytl.n3l_list,
            "k1": self.p1.k1l,
            "k2": self.p2.k2l,
            "c": self.phi.C_phil,
            "lifs_strongest": list(self.p1.strongest.index),
            "lifs_strongest_xpol": list(self.p1.strongest_xpol.index),
            "lifs_other": list(self.p2.other.index),
        }

    def log_verbose(self):
        logger.info(self.vm1m2)
        logger.info(self.ytl)
        logger.info(f"{self.i17}")
        logger.info(f"{self.i23}")
        logger.info(f"{self.i24}")
        logger.info(f"{self.i25}")
        logger.info(f"{self.i18}")
        logger.info(self.p1)
        logger.info(self.p2)
        logger.info(self.phi)
        logger.info(self.yp2phitl)
        logger.info(self.p1yp2phitl)
        logger.info(f"Gamma matrix:\n{self.p1yp2phitl.gamma_matrix}")
        logger.info(self.w)

    def log_summary(self):
        logger.info(f"Coefficients: {self.coefs}")
        logger.info(f"Beam settings: {self.beams}")
        logger.info(f"Coefficients bin: {self.coefs_bin}")
        logger.info(f"Coefficients str: {self.coefs_str}")


class PmiGenerator:
    def __init__(self, comb_inx: int, R: int, N3: int, v: int, N1N2: tuple[int, int], **kwargs):
        self.comb_inx: int = comb_inx
        self.R: int = R
        self.N3: int = N3
        self.v: int = v
        self.N1N2: tuple[int, int] = N1N2
        self.O1O2: tuple[int, int] = PcsirsN1N2.get_O1O2(self.N1N2)
        self.Pcsirs: int = PcsirsN1N2.get_Pcsirs(self.N1N2)
        self.L, self.pv, self.beta = ParamCombR16.l_pv_beta(comb_inx=comb_inx, v=v)
        self.M_v = ParamCombR16.M_v(comb_inx=comb_inx, R=R, N3=N3, v=v)
        self.K0 = ParamCombR16.K0(comb_inx=comb_inx, R=R, N3=N3, v=v)

    def factory(self, i11_or_q1q2, i12_or_n1n2, i15, i16, i17, i23, i24, i25, i18) -> Pmi:
        i11_obj = I11(i11_or_q1q2=i11_or_q1q2, O1O2=self.O1O2)
        i12_obj = I12(i12_or_n1n2=i12_or_n1n2, N1N2=self.N1N2, L=self.L, rel16=True)
        vm1m2 = Vm1m2(i11=i11_obj.as_int, i12=i12_obj.as_int, N1N2=self.N1N2, O1O2=self.O1O2, L=self.L, rel16=True)
        y = Y(Yl.factory(comb_inx=self.comb_inx, R=self.R, N3=self.N3, v=self.v, i16=i16, i15=i15))
        i17_obj = I17l.factory(i17=i17, n3l=y.n3l_list, L=self.L, N3=self.N3, v=self.v, K0=self.K0)
        i23_obj = I23l(v=self.v, i23=i23)
        i24_obj = I24l(v=self.v, K_NZ=i17_obj.K_NZ, i24=i24)
        i25_obj = I25l(v=self.v, K_NZ=i17_obj.K_NZ, i25=i25)
        i18_obj = I18l(L=self.L, v=self.v, K_NZ=i17_obj.K_NZ, N3=self.N3, i18=i18)
        p1 = P1(i23=i23_obj, i17=i17_obj, i18=i18_obj)
        p2 = P2(i17=i17_obj, i18=i18_obj, i24=i24_obj)
        phi = Phi(i17=i17_obj, i18=i18_obj, i25=i25_obj)
        yp2phitl = YP2Phitl(ytl=y, p2=p2, phi=phi)
        p1yp2phitl = P1Yp2Phitl(yp2phitl=yp2phitl, p1=p1)
        w = W(vm1m2=vm1m2, p1yp2phitl=p1yp2phitl)
        return Pmi(w=w, i11=i11_obj, i12=i12_obj, i17=i17_obj, i18=i18_obj, i23=i23_obj, i24=i24_obj, i25=i25_obj)

    def beam_factory(self, n1n2: tuple[list[int], list[int]], q1q2: tuple[int, int], n3l: list[list[int]],
                     lifs_strongest: list[tuple[int, int, int]], lifs_other: list[tuple[int, int, int]],
                     k1: list[int], k2: list[int], c: list[int], i15: int = 0, **kwargs) -> Pmi:
        return self.factory(i11_or_q1q2=q1q2, i12_or_n1n2=n1n2, i15=i15, i16=n3l,
                            i17=list(set(lifs_strongest + lifs_other)),
                            i18=[i for l, i, f in sorted(lifs_strongest)],
                            i23=k1, i24=k2, i25=c)

    def coef_factory(self, i11: int | str, i12: int | str, i16: int | str,
                     i17: str, i18: str, i23: str, i24: str, i25: str, i15: int = 0, **kwargs) -> Pmi:
        return self.factory(i11_or_q1q2=i11, i12_or_n1n2=i12, i15=i15, i16=i16,
                            i17=i17, i18=i18, i23=i23, i24=i24, i25=i25)


def example_coef_factory_test_v4(verbose: bool = True) -> Pmi:
    """Example function showcasing precoding matrix calculation for the given PMI coefficients
    """
    coefs = {
            "i11": "0100",
            "i12": "0001000",
            "i18": "001010011011",
            "i23": "0001110001010111",
            "i15": "",
            "i16": "100101001010100101001010100101001010100101001010",
            "i24": "011010010100",
            "i25": "1111111110111000",
            "i17": "1111100001000011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        }

    gen = PmiGenerator(comb_inx=6, R=1, N3=18, N1N2=(4, 2), v=4)
    pmi = gen.coef_factory(**coefs)
    if verbose:
        pmi.log_verbose()
    pmi.log_summary()
    return pmi


def example_beam_factory_test_v1(verbose: bool = True) -> Pmi:
    q1q2 = (0, 1)
    n1n2 = ([2, 0, 1, 2], [1, 2, 2, 2])
    n3l = [[0, 1, 2, 3, 5]]
    lifs_strongest = [(1, 1, 0)]
    lifs_other = [(1, 0, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4)]
    k1l = [13]
    k2l = [5, 1, 3, 4]
    C_fil = [11, 10, 15, 12]

    gen = PmiGenerator(comb_inx=5, R=1, N3=18, v=1, N1N2=(4,3))
    pmi = gen.beam_factory(n1n2=n1n2, q1q2=q1q2, n3l=n3l,
                           lifs_strongest=lifs_strongest, lifs_other=lifs_other,
                           k1=k1l, k2=k2l, c=C_fil, i15=0)
    if verbose:
        pmi.log_verbose()
    pmi.log_summary()
    return pmi


def example_pmi_generator(ant_dim: tuple[int, int] | None = None, comb_inx: int = 6, N3: int = 35, R: int = 2,
                          v: int = 4, optimize_rank: bool = True, randomize: bool = False, verbose: bool = False) -> Pmi:
    lifs_strongest = {
        (1, True): [(1, 1, 0)],
        (2, True): [(1, 1, 0), (2, 2, 0)],
        (3, True): [(1, 1, 0), (2, 1, 0), (3, 1, 0)],
        (4, True): [(1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0)],
        (2, False): [(1, 1, 0), (2, 1, 0)],
        (3, False): [(1, 1, 0), (2, 1, 0), (3, 1, 0)],
        (4, False): [(1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0)],
    }
    lifs_other = {
        (1, True): [(1, 0, 0), (1, 2, 2)],
        (2, True): [(1, 0, 0), (2, 0, 2), (2, 1, 3), (1, 2, 3)],
        (3, True): [(1, 0, 0), (2, 0, 1), (3, 0, 2), (1, 2, 2)],
        (4, True): [(1, 0, 1), (2, 0, 1), (3, 0, 2), (4, 0, 2)],
        (2, False): [(1, 0, 0), (2, 0, 2), (2, 2, 3), (1, 2, 3)],
        (3, False): [(1, 0, 0), (2, 0, 2), (3, 0, 2)],
        (4, False): [(1, 0, 1), (2, 0, 2), (3, 0, 2), (4, 0, 2)],
    }
    k1 = {
        (1, True): [6],
        (2, True): [4, 6],
        (3, True): [3, 2, 15],
        (4, True): [1, 12, 5, 7],
        (2, False): [4, 4],
        (3, False): [5, 5, 5],
        (4, False): [12, 12, 5, 5],
    }
    k2 = {
        (1, True): [5, 4],
        (2, True): [0, 5, 4, 1],
        (3, True): [1, 4, 2, 3],
        (4, True): [3, 2, 2, 4],
        (2, False): [0, 0, 4, 4],
        (3, False): [4, 4, 4],
        (4, False): [3, 3, 2, 2],
    }
    c = {
        (1, True): [10, 5],
        (2, True): [10, 5, 6, 0],
        (3, True): [0, 3, 7, 14],
        (4, True): [15, 15, 11, 8],
        (2, False): [10, 10, 6, 6],
        (3, False): [10, 10, 10],
        (4, False): [15, 15, 8, 8],
    }

    ant_dim = ant_dim or (4, 2)
    logger.info(f"PMI generation for ant_dim:{ant_dim}, comb_inx:{comb_inx}, R:{R}, N3:{N3}, v:{v} "
                f"(optimize_rank:{optimize_rank}, randomize:{randomize})")
    i15_n3l_options = Yl.n3l_generator(comb_inx=comb_inx, R=R, N3=N3, v=v, randomize=randomize)
    opt_inx = int(len(i15_n3l_options.keys()) / 2)
    i15 = i15_n3l_options.get(opt_inx).get("i15")
    n3l = [i15_n3l_options.get(opt_inx).get("n3l")] * v

    beam_settings = {'q1q2': (1, 0), 'n1n2': ([2, 0, 1, 2], [0, 1, 1, 1]),
                     'n3l': n3l, 'i15': i15,
                     'k1':  k1.get((v, optimize_rank), k1.get((1, True))),
                     'k2': k2.get((v, optimize_rank),  k2.get((1, True))),
                     'c': c.get((v, optimize_rank), c.get((1, True))),
                     'lifs_strongest': lifs_strongest.get((v, optimize_rank), lifs_strongest.get((1, True))),
                     'lifs_other': lifs_other.get((v, optimize_rank), lifs_other.get((1, True)))}

    gen = PmiGenerator(comb_inx=comb_inx, R=R, N3=N3, v=v, N1N2=ant_dim)
    pmi = gen.beam_factory(**beam_settings)
    if verbose:
        pmi.log_verbose()
    pmi.log_summary()
    return pmi


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    #pmi = example_coef_factory_test_v4()
    #pmi = example_beam_factory_test_v1()
    pmi = example_pmi_generator(ant_dim=(4, 2), comb_inx=6, N3=18, R=1, v=4)
    pass