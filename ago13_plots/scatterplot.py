from matplotlib import pyplot as plt

# the relationship between the number of friends your users have
# and the number of minutes they spend on the site every day

friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)
for label, friend_count, minute_count in (
        zip(labels, friends, minutes)):
    plt.annotate(label, xy = (friend_count, minute_count),
                 xytext=(5, -5), textcoords='offset points')
plt.title("Daily minutes vs. Number of friends")
plt.xlabel("Num. of friends")
plt.ylabel("Daily minutes")
plt.show()
