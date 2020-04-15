def gen(n, counter_open=0, counter_close=0, ans=''):
    if counter_open + counter_close == 2 * n:
        print(ans)
        return
    if counter_open < n:
        gen(n, counter_open + 1, counter_close, ans + '(')
    if counter_open > counter_close:
        gen(n, counter_open, counter_close + 1, ans + ')')

gen(int(input()))