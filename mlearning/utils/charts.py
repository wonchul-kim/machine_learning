import matplotlib.pyplot as plt
import seaborn as sns


def autopct_with_counts(pct, allvals):
    absolute = int(pct/100.*sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def draw_pie_chart(data, title, output_filename):
   sns.set_theme(style="whitegrid")
   plt.figure(figsize=(6, 6))
   plt.suptitle(title + f' ({sum(data.values())})')
   plt.pie(data.values(), labels=data.keys(), autopct=lambda pct: autopct_with_counts(pct, list(data.values())), startangle=140, colors=sns.color_palette("pastel"))
   plt.savefig(output_filename)
   
   
def draw_dist_chart(data, key, output_filename):
    plt.figure(figsize=(12, 10)) 
    plt.subplot(2, 1, 1)
    plt.hist(list(map(int, data['width'])), bins=range(min(list(map(int, data['width']))), max(list(map(int, data['width']))) + 2), edgecolor='black', alpha=0.7)
    # plt.xticks(range(min(list(map(int, data['width']))), max(list(map(int, data['width']))) + 1)) 
    plt.xlabel('width') 
    plt.ylabel('Frequency')
    plt.subplot(2, 1, 2)
    plt.hist(list(map(int, data['height'])), bins=range(min(list(map(int, data['height']))), max(list(map(int, data['height']))) + 2), edgecolor='black', alpha=0.7)
    # plt.xticks(range(min(list(map(int, data['height']))), max(list(map(int, data['height']))) + 1)) 
    plt.suptitle(f'Distribution of {key}') 
    plt.xlabel('height') 
    plt.ylabel('Frequency')
    plt.grid(True)  
    plt.savefig(output_filename)