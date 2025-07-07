import numpy as np
import math
import numpy.typing as npt
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class PcsirsN1N2:
    """
    38.214 Table 5.2.2.2.1-2
    """
    df = pd.DataFrame([[4,  (2,1), (4,1)],
                        [8,  (2,2), (4,4)],
                        [8,  (4,1), (4,1)],
                        [12, (3,2), (4,4)],
                        [12, (6,1), (4,1)],
                        [16, (4,2), (4,4)],
                        [16, (8,1), (4,1)],
                        [24, (4,3), (4,4)],
                        [24, (6,2), (4,4)],
                        [24, (12,1),(4,1)],
                        [32, (4,4), (4,4)],
                        [32, (8,2), (4,4)],
                        [32, (16,1),(4,1)]], columns=["Pcsirs", "N1N2", "O1O2"])

    @classmethod
    def get_row(cls, n1n2: tuple[int, int]) -> pd.DataFrame:
        return cls.df.loc[cls.df["N1N2"] == n1n2]

    @classmethod
    def get_O1O2(cls, n1n2: tuple[int, int]) -> tuple[int, int]:
        _vals = cls.get_row(n1n2=n1n2).get("O1O2").values
        return _vals[-1] if _vals.size else tuple()

    @classmethod
    def get_Pcsirs(cls, n1n2: tuple[int, int]) -> int:
        _vals = cls.get_row(n1n2=n1n2).get("Pcsirs").values
        return int(_vals[-1]) if _vals.size else 0


class Base:
    def __init__(self):
        self._size: int = 0
        self._value: str = ""

    @property
    def size(self) -> int:
        return self._size

    @property
    def as_bin(self) -> str:
        return self._value

    @as_bin.setter
    def as_bin(self, v: str | int):
        self._value = self.to_bin_str(v=v, size=self.size)

    @property
    def as_hex(self) -> str:
        return hex(self.as_int) if self.as_int is not None else ""

    @as_hex.setter
    def as_hex(self, v: str | int):
        self._value = self.to_bin_str(v=v, size=self.size)

    @property
    def as_int(self) -> int | None:
        return int(self.as_bin, 2) if self.as_bin else None

    @as_int.setter
    def as_int(self, v: str | int):
        self._value = self.to_bin_str(v=v, size=self.size)

    # @classmethod
    # def bin_str(cls, value: int, size: int) -> str:
    #    return f"{value:#0{size+2}b}" if value is not None else ""

    @classmethod
    def to_bin_str(cls, v: str | int, size: int=0) -> str:
        if isinstance(v, str):
            v = cls.strip_spaces(v)
            if v.startswith("0x"):
                v = bin(int(v, 16))[2:]
            elif v.startswith("0b"):
                v = v[2:]
        elif isinstance(v, (int, np.integer)):
            v = bin(v)[2:]
        else:
            v = ""
        if size:
            v = v.zfill(size)
            v = v[:size]
        return v

    @classmethod
    def align_bit_length(cls, v: int, size: int, factor: int = 8) -> int:
        _diff = size % factor
        return v << (factor - _diff) if _diff else v

    @classmethod
    def bit_size_log(cls, n: int, k: int = 1):
        return math.ceil(math.log2(math.comb(n, k) if n >= k else 0))

    @classmethod
    def strip_spaces(cls, v: str) -> str:
        return v.replace(" ", "")


class BaseMatrix:
    def __init__(self, shape: tuple[int, ...], dtype: npt.DTypeLike, vsplit: int = None, shape_desc: str = ""):
        self._shape: tuple[int, ...] = shape
        self._shape_desc: str = f" ({shape_desc})" or ""
        self._dtype: npt.DTypeLike = dtype
        self._vsplit: int | None = vsplit or 1
        self._matrix: npt.NDArray = np.zeros(shape, dtype=dtype)

    def __str__(self):
        return f"{self.__class__}: matrix {self.shape}{self._shape_desc}:\n{self.matrix}"

    @property
    def matrix(self) -> npt.NDArray:
        return self._matrix

    @property
    def matrix_abs(self) -> npt.NDArray:
        return np.absolute(self._matrix)

    @property
    def matrix_T(self) -> npt.NDArray:
        return self._matrix.T

    @property
    def matrices(self) -> list[npt.NDArray]:
        return np.vsplit(self.matrix, self._vsplit)

    @property
    def shape(self) -> tuple[int,...]:
        return self.matrix.shape

    @property
    def shape_desc(self) -> str:
        return self._shape_desc

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.matrix.dtype

    @property
    def vsplit(self) -> int:
        return self._vsplit

    def for_l(self, l: int) -> npt.NDArray:
        return self.matrices[l - 1]


class I11(Base):
    def __init__(self, i11_or_q1q2: tuple[int, int] | int | str, O1O2: tuple[int, int]):
        super().__init__()
        self._o1 = O1O2[0]
        self._o2 = O1O2[1]
        self._size = self.bit_size_log(self._o1 * self._o2)
        self._size_o1 = self.bit_size_log(self._o1)
        self._size_o2 = self.bit_size_log(self._o2)
        self._value = self._init(i11_or_q1q2)
        self._check()

    def _init(self, v: tuple[int, int] | int | str) -> str:
        _v = v
        if isinstance(v, tuple):
            q1, q2 = v
            assert q1 < (1 << self._size_o1), f"q1:{q1} > O1:{self._o1}"
            assert q2 < (1 << self._size_o2), f"q2:{q2} > O2:{self._o2}"
            _v = q2 + (q1 << self._size_o2)
        return self.to_bin_str(_v, self._size)

    def _check(self):
        if self.as_bin:
            assert self.as_int < (1 << self._size), f"i11:{self.as_int} > O1*O2:{self._o1 * self._o2}"

    def __str__(self):
        return f"{self.__class__}: q1={self.q1}, q2={self.q2}, i11:{self.as_int} ({self.as_bin})"

    @property
    def q2(self) -> int:
        return self.as_int & ((1 << self._size_o2) - 1) if self.as_bin else None

    @property
    def q1(self) -> int:
        return self.as_int >> self._size_o2 if self.as_bin else None

    @property
    def q1q2(self) -> tuple[int, int]:
        return self.q1, self.q2

    @classmethod
    def get_bit_width(cls, O1O2: tuple[int, int]) -> int:
        return cls.bit_size_log(O1O2[0] * O1O2[1])


class I12(Base):
    # 38.214 Table 5.2.2.2.3-1
    C_table = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 3, 1, 0],
        [4, 6, 4, 1],
        [5, 10, 10, 5],
        [6, 15, 20, 15],
        [7, 21, 35, 35],
        [8, 28, 56, 70],
        [9, 36, 84, 126],
        [10, 45, 120, 210],
        [11, 55, 165, 330],
        [12, 66, 220, 495],
        [13, 78, 286, 715],
        [14, 91, 364, 1001],
        [15, 105, 455, 1365]
    ])
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

    def __init__(self, i12_or_n1n2: tuple[list[int], list[int]] | int | str, N1N2: tuple[int, int], L: int,
                 rel16: bool = False):
        super().__init__()
        self._rel16 = rel16
        self._N1 = N1N2[0]
        self._N2 = N1N2[1]
        self._L = L
        self._size = self.bit_size_log(self.N1 * self.N2, L)
        if isinstance(i12_or_n1n2, tuple):
            self.as_int = self.get_i12(n1=i12_or_n1n2[0], n2=i12_or_n1n2[1], N1=self.N1, N2=self.N2, L=L,
                                    rel16=self._rel16)
            self._n1n2 = i12_or_n1n2
        else:
            self.as_int = i12_or_n1n2
            self._n1n2 = self.get_n1n2(i12=self.as_int, N1=self.N1, N2=self.N2, L=L, rel16=self._rel16)
        self._check_consistency()

    def _check_consistency(self):
        _i12 = self.get_i12(n1=self.n1n2[0], n2=self.n1n2[1], N1=self.N1, N2=self.N2, L=self._L, rel16=self._rel16)
        assert self.as_int == _i12, f"Inconsistent settings (i12:{self.as_int} =! i12_from_n1n2:{_i12}"
        _n1n2 = np.array(self.n1n2)
        assert np.unique(_n1n2, axis=1).shape[-1] == _n1n2.shape[-1], \
            f"Inconsistent settings (n1, n2 coordinates are not unique: {_n1n2})"

    def __str__(self):
        return f"{self.__class__}: n1={self.n1}, n2={self.n2}, i12:{self.as_int} ({self.as_bin})"

    @property
    def n1n2(self) -> tuple[list[int], list[int]]:
        return self._n1n2

    @property
    def n1(self) -> list[int]:
        return self.n1n2[0]

    @property
    def n2(self) -> list[int]:
        return self.n1n2[-1]

    @property
    def N1(self) -> int:
        return self._N1

    @property
    def N2(self) -> int:
        return self._N2

    @classmethod
    def _check_n1n2(cls, n1: list[int], n2: list[int], N1: int, N2: int, L: int):
        assert len(n1) == L, f"Wrong n1 size ({len(n1)}, L:{L})"
        assert len(n2) == L, f"Wrong n2 size ({len(n2)}, L:{L})"
        assert all(map(lambda num: 0 <= num < N1, n1)), f"Some n1 values ({n1}) out of range ({range(N1)})"
        assert all(map(lambda num: 0 <= num < N2, n2)), f"Some n2 values ({n2}) out of range ({range(N2)})"
        if N2 == 1 and any(map(lambda v: 0 != v, n2)):
            raise AssertionError(f"If N2==1, n2 shall be zeros and not {n2} (TS38.214 sec. 5.2.2.2.3)")

    @classmethod
    def get_i12(cls, n1: list[int], n2: list[int], N1: int = 4, N2: int = 1,
                L: int = 2, rel16: bool = False) -> int | None:
        cls._check_n1n2(n1=n1, n2=n2, N1=N1, N2=N2, L=L)
        if ((N1, N2) == (2, 1) and n1 == [2, 1] and n2 == [0, 0]) \
                or ((N1, N2) == (4, 1) and L == 4 and set(n1) == {0, 1, 2, 3} and set(n2) == {0, 0, 0, 0}) \
                or ((N1, N2) == (2, 2) and L == 4 and set(n1) == {0, 1, 0, 1} and set(n2) == {0, 0, 1, 1}):
            logger.info(f"i12 is not reported")
            return None

        n = np.sort(np.array(n2) * N1 + np.array(n1))
        logger.debug(f"n^(i): {n}")
        i12 = 0
        C = cls.c_tab(is_rel16=rel16)
        for i, n_i in enumerate(list(n)):
            x = N1 * N2 - 1 - n_i
            y = L - 1 - i
            logger.debug(f"Searching i12, iter#{i}: (x, y): ({x}, {y}), C(x,y): {C[x, y]}")
            i12 += C[x, y]
        return i12

    @classmethod
    def get_n1n2(cls, i12: int, N1: int = 4, N2: int = 1, L: int = 2, rel16: bool = False):
        s_prev = 0
        n1: list[int] = []
        n2: list[int] = []
        C = cls.c_tab(is_rel16=rel16)
        for i in range(L):
            sorter = np.argsort(C[:, L - 1 - i])
            # old implementation did not consider the case that the x_star could theoretically be lower than next index,
            # however on the other hand that should never happen, because the max possible i12 depends on N1, N2 and L
            # and so the new implementation might be superfluous
            #x_star = np.searchsorted(C[:, L-1-i], i12-s_prev, side="right", sorter=sorter) - 1
            x_star_upper = np.searchsorted(C[:, L - 1 - i], i12 - s_prev, side="right", sorter=sorter) # index of the 1st larger elem
            x_star_arr = np.array(range(x_star_upper))
            x_star_allowed = np.where(((L - 1 - i) <= x_star_arr) & (x_star_arr < (N1 * N2 - i)))
            x_star = x_star_allowed[0].max() if x_star_allowed[0].size else -1
            assert L - 1 - i <= x_star < N1 * N2 - i, f"X_star {x_star} out of allowed range ({L - 1 - i}, {N1 * N2 - i - 1}). Check i12 value!"
            e_i = C[x_star, L - 1 - i]
            s_prev = s_prev + e_i
            n_i = N1 * N2 - 1 - x_star
            n1.append(int(n_i % N1))
            n2.append(int((n_i - n1[-1]) / N1))
            logger.debug(f"find n1,n2, iter#{i}: x_star:{x_star}, e_i:{e_i}, n_i: {n_i}, n1: {n1[-1]}, n2: {n2[-1]}")
        return n1, n2

    @classmethod
    def c_tab(cls, is_rel16: bool = False):
        return cls.C_table_r16 if is_rel16 else cls.C_table

    @classmethod
    def get_bit_width(cls, N1N2: tuple[int, int], L: int) -> int:
        return cls.bit_size_log(N1N2[0] * N1N2[1], L)


class Vm1m2:
    def __init__(self, i11: int | str, i12: int | str, N1N2: tuple[int, int],
                 O1O2: tuple[int, int], L: int = 2, rel16: bool = False):
        N1, N2 = N1N2
        O1, O2 = O1O2
        self._i11: I11 = I11(i11_or_q1q2=i11, O1O2=O1O2)
        self._i12: I12 = I12(i12_or_n1n2=i12, N1N2=N1N2, L=L, rel16=rel16)
        self._m1: list[int] = list(O1 * np.array(self.i12.n1) + self.i11.q1)
        self._m2: list[int] = list(O2 * np.array(self.i12.n2) + self.i11.q2)
        self._um: npt.NDArray[np.complex128] = np.array([self.um_i(N2, O2, m) for m in self._m2])
        self._vlm: npt.NDArray[np.complex128] = np.array([self.vlm_i(list(self._um[i]), N1, O1, l)
                                                          for i, l in enumerate(self._m1)])

    def __str__(self):
        return f"{self.__class__}: m1={self.m1}, m2={self.m2}\n{self.vlm_w1}"

    @property
    def i12(self) -> I12:
        return self._i12

    @property
    def i11(self) -> I11:
        return self._i11

    @property
    def m1(self):
        return self._m1

    @property
    def m2(self):
        return self._m2

    @property
    def m1m2(self) -> tuple[list[int], list[int]]:
        return self.m1, self.m2

    @property
    def vlm(self) -> npt.NDArray[np.complex128]:
        return self._vlm.T

    @property
    def vlm_w1(self) -> npt.NDArray[np.complex128]:
        return np.kron(np.eye(2), self._vlm.T)

    @classmethod
    def um_i(cls, N2: int, O2: int, m: int) -> list[np.complex128]:
        return [np.around(np.exp(1j * 2 * np.pi * m * i / (N2 * O2)), 3) for i in range(N2)]

    @classmethod
    def vlm_i(cls, um: list[np.complex128], N1: int, O1: int, l: int) -> list[np.complex128]:
        vl = np.array([np.exp(1j * 2 * np.pi * l * i / (N1 * O1)) for i in range(N1)])
        # logger.info(f"vl: {vl}")
        return list(np.around(np.kron(vl, np.array(um)), 3))