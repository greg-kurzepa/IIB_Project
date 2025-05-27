#%%

def func(a):
    class A:
        def __init__(self):
            pass

        def Afunc(self):
            print(a)

    return A()

dict1 = {'a': 1, 'b': 2}
instance = func(dict1)
instance.Afunc()
# %%

def func_inner(a, b):
    return a + b

def func2(fix_b):
    def func_fix(a):
        return func_inner(a, fix_b)
    
    return func_fix

print(func_inner(5,3))
f = func2(9)
print(f(2))
# %%
