# Testing

# Stacked Bar Diagram 

x = ['A','B','C','D']
y1 = [10,21,12,30]
y2 = [15,32,12,23]

plt.bar(x,y1)
plt.bar(x,y2, bottom=y1)

plt.title("Stacked Bar Diagram")
plt.xlabel("Name")
plt.ylabel("Age")

plt.show()
