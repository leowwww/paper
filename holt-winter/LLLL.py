def h(x,y,a):
    ans=0.0
    for i in range(len(y)):
        t=y[i]
        for j in range(len(y)):
            if i !=j:
                t*=(a-x[j])/(x[i]-x[j])
        ans +=t
    return ans
x=[50,40,30,20,10]
y=[-4.5,6.5,13.6,18.2,21]
print(h(x ,y,10))
for i in range(-100000,100000):
    if h(x,y,i) == 100:
        print(i)
        break
#print(h(x,y,2))
