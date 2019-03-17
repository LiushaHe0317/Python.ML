
def continuous_cum(arr):
    
    if max(arr) < 0:
        s = arr[0]
    elif max(arr) == 0:
        s = max(arr)
    elif min(arr) < 0:
        s = sum(arr)
    else:
        current_sum = 0
        sum_set = []
        
        for i in arr:
            current_sum = current_sum + i
            sum_set.append(current_sum)
            
            if current_sum < 0:
                current_sum = 0
        
        s = max(sum_set)
        
    return s

arr = [2,11,10,9,2]
s = continuous_cum(arr)
print(s)
