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
