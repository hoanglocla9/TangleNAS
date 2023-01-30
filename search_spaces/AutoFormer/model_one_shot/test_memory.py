import torch


def weighted_sum(l):
    s = torch.zeros([32, 3, 32, 32]).cuda()
    for a in l:
        s = s + a
    return s


#li = [a,b,c]
#s1 = weighted_sum(li)
#print(torch.cuda.memory_allocated())
a = torch.randn([32, 3, 32, 32]).cuda()
b = torch.randn([32, 3, 32, 32]).cuda()
#a = a+b
c = torch.randn([32, 3, 32, 32]).cuda()
#a = a+c
li = [a, b, c]
weighted_sum(li)
#print(torch.sum(s==s1))
print(torch.cuda.memory_allocated())
