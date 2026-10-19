"""补强项 3 用到的独立 oracle 库：每道题一个按题意直接计算的实现，加上合法输入的生成器和校验器。

- generate(rng)：生成一个合法的小输入，参数名与题目签名一致，故意多覆盖边界。
- validate(**kwargs)：按题目约束检查输入是否合法，不合法的候选直接标 invalid。
- oracle(**kwargs)：按题意直接计算答案，越直白越好，只用在小输入上，写法要和数据集参考解不同。

只有参考解与 oracle 一致的用例才会被接受。这里收录 7 道示范题，覆盖三种情况：
测试确实太弱；变异体存活但其实是等价改动；杀死率满分却仍漏掉离参考解很远的错解。
要扩到更多题，就照这个格式继续加。
"""
import itertools


def _ints(rng, low, high, min_len, max_len):
    return [rng.randint(low, high) for _ in range(rng.randint(min_len, max_len))]


def _int_range(values, low, high):
    return all(type(value) is int and low <= value <= high for value in values)


# LeetCode 53 最大子数组和：枚举全部非空连续子数组。
def _max_subarray_oracle(nums):
    return max(sum(nums[i:j + 1]) for i in range(len(nums)) for j in range(i, len(nums)))


def _max_subarray_generate(rng):
    return {"nums": _ints(rng, -10, 10, 1, 8)}


def _max_subarray_validate(nums):
    return 1 <= len(nums) <= 10**5 and _int_range(nums, -10**4, 10**4)


# LeetCode 334 递增的三元子序列：枚举所有下标三元组。
def _triplet_oracle(nums):
    return any(nums[i] < nums[j] < nums[k] for i, j, k in itertools.combinations(range(len(nums)), 3))


def _triplet_generate(rng):
    return {"nums": _ints(rng, -3, 3, 1, 8)}  # 值域小，重复值多，专门考严格小于


def _triplet_validate(nums):
    return 1 <= len(nums) <= 5 * 10**5 and _int_range(nums, -2**31, 2**31 - 1)


# LeetCode 125 验证回文串：只留字母数字、统一小写后与反转比较。
def _palindrome_oracle(s):
    kept = [ch.lower() for ch in s if ch.isalnum()]
    return kept == kept[::-1]


def _palindrome_generate(rng):
    text = "".join(rng.choice("abAB12 ,.:") for _ in range(rng.randint(1, 8)))
    if rng.random() < 0.5:
        text += text[::-1]  # 一半候选做成镜像，保证有足够多的回文
    return {"s": text}


def _palindrome_validate(s):
    return 1 <= len(s) <= 2 * 10**5 and all(32 <= ord(ch) <= 126 for ch in s)


# LeetCode 2678 老人的数目：年龄是第 12、13 位数字，严格大于 60 才算。
def _seniors_oracle(details):
    return sum(1 for item in details if int(item[11]) * 10 + int(item[12]) > 60)


def _seniors_generate(rng):
    rows, phones, seats = [], set(), set()
    while len(rows) < rng.randint(1, 5):
        phone = "".join(rng.choice("0123456789") for _ in range(10))
        seat = f"{rng.randint(0, 99):02d}"
        if phone in phones or seat in seats:
            continue
        phones.add(phone)
        seats.add(seat)
        age = rng.choice([rng.randint(0, 99), 59, 60, 61])  # 故意多给边界年龄
        rows.append(f"{phone}{rng.choice('MFO')}{age:02d}{seat}")
    return {"details": rows}


def _seniors_validate(details):
    return (1 <= len(details) <= 100
            and all(len(item) == 15 and item[:10].isdigit() and item[10] in "MFO" and item[11:].isdigit()
                    for item in details)
            and len({item[:10] for item in details}) == len(details)
            and len({item[13:] for item in details}) == len(details))


# LeetCode 153 寻找旋转排序数组中的最小值：直接取最小值。
def _rotated_min_oracle(nums):
    return min(nums)


def _rotated_min_generate(rng):
    size = rng.randint(1, 8)
    base = sorted(rng.sample(range(-20, 21), size))
    shift = rng.randint(0, size - 1)
    return {"nums": base[shift:] + base[:shift]}


def _rotated_min_validate(nums):
    size = len(nums)
    if not (1 <= size <= 5000 and len(set(nums)) == size and _int_range(nums, -5000, 5000)):
        return False
    return any(nums[k:] + nums[:k] == sorted(nums) for k in range(size))  # 必须是有序数组旋转来的


# LeetCode 414 第三大的数：去重排序后按题意取值。
def _third_max_oracle(nums):
    distinct = sorted(set(nums), reverse=True)
    return distinct[2] if len(distinct) >= 3 else distinct[0]


def _third_max_generate(rng):
    return {"nums": _ints(rng, -3, 3, 1, 7)}


def _third_max_validate(nums):
    return 1 <= len(nums) <= 10**4 and _int_range(nums, -2**31, 2**31 - 1)


# LeetCode 1470 重新排列数组：前后两半交错。
def _shuffle_oracle(nums, n):
    return [value for pair in zip(nums[:n], nums[n:]) for value in pair]


def _shuffle_generate(rng):
    half = rng.randint(1, 4)
    return {"nums": [rng.randint(1, 1000) for _ in range(2 * half)], "n": half}


def _shuffle_validate(nums, n):
    return 1 <= n <= 500 and len(nums) == 2 * n and _int_range(nums, 1, 1000)


ORACLES = {
    "maximum-subarray": {"generate": _max_subarray_generate, "validate": _max_subarray_validate,
                         "oracle": _max_subarray_oracle},
    "increasing-triplet-subsequence": {"generate": _triplet_generate, "validate": _triplet_validate,
                                       "oracle": _triplet_oracle},
    "valid-palindrome": {"generate": _palindrome_generate, "validate": _palindrome_validate,
                         "oracle": _palindrome_oracle},
    "number-of-senior-citizens": {"generate": _seniors_generate, "validate": _seniors_validate,
                                  "oracle": _seniors_oracle},
    "find-minimum-in-rotated-sorted-array": {"generate": _rotated_min_generate, "validate": _rotated_min_validate,
                                             "oracle": _rotated_min_oracle},
    "third-maximum-number": {"generate": _third_max_generate, "validate": _third_max_validate,
                             "oracle": _third_max_oracle},
    "shuffle-the-array": {"generate": _shuffle_generate, "validate": _shuffle_validate,
                          "oracle": _shuffle_oracle},
}
