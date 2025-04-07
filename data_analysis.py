import csv
import statistics as st

age_1 = []
bmi_1 = []
glucose_1 = []
insulin_1 = []
homa_1 = []
leptin_1 = []
adiponectin_1 = []
resistin_1 = []
mcp1_1 = []

age_2 = []
bmi_2 = []
glucose_2 = []
insulin_2 = []
homa_2 = []
leptin_2 = []
adiponectin_2 = []
resistin_2 = []
mcp1_2 = []

with open(r'Breast Cancer Coimbra_MLR\dataR2.csv', newline='') as data:
    reader = csv.reader(data, delimiter=',')
    for row in reader:
        if row[9] == "1":
            age_1.append(float(row[0]))
            bmi_1.append(float(row[1]))
            glucose_1.append(float(row[2]))
            insulin_1.append(float(row[3]))
            homa_1.append(float(row[4]))
            leptin_1.append(float(row[5]))
            adiponectin_1.append(float(row[6]))
            resistin_1.append(float(row[7]))
            mcp1_1.append(float(row[8]))
        elif row[9] == "2":
            age_2.append(float(row[0]))
            bmi_2.append(float(row[1]))
            glucose_2.append(float(row[2]))
            insulin_2.append(float(row[3]))
            homa_2.append(float(row[4]))
            leptin_2.append(float(row[5]))
            adiponectin_2.append(float(row[6]))
            resistin_2.append(float(row[7]))
            mcp1_2.append(float(row[8]))

age_1_mean = st.mean(age_1)
bmi_1_mean = st.mean(bmi_1)
glucose_1_mean = st.mean(glucose_1)
insulin_1_mean = st.mean(insulin_1)
homa_1_mean = st.mean(homa_1)
leptin_1_mean = st.mean(leptin_1)
adiponectin_1_mean = st.mean(adiponectin_1)
resistin_1_mean = st.mean(resistin_1)
mcp1_1_mean = st.mean(mcp1_1)

age_2_mean = st.mean(age_2)
bmi_2_mean = st.mean(bmi_2)
glucose_2_mean = st.mean(glucose_2)
insulin_2_mean = st.mean(insulin_2)
homa_2_mean = st.mean(homa_2)
leptin_2_mean = st.mean(leptin_2)
adiponectin_2_mean = st.mean(adiponectin_2)
resistin_2_mean = st.mean(resistin_2)
mcp1_2_mean = st.mean(mcp1_2)

age_1_std = st.stdev(age_1)
bmi_1_std = st.stdev(bmi_1)
glucose_1_std = st.stdev(glucose_1)
insulin_1_std = st.stdev(insulin_1)
homa_1_std = st.stdev(homa_1)
leptin_1_std = st.stdev(leptin_1)
adiponectin_1_std = st.stdev(adiponectin_1)
resistin_1_std = st.stdev(resistin_1)
mcp1_1_std = st.stdev(mcp1_1)

age_2_std = st.stdev(age_2)
bmi_2_std = st.stdev(bmi_2)
glucose_2_std = st.stdev(glucose_2)
insulin_2_std = st.stdev(insulin_2)
homa_2_std = st.stdev(homa_2)
leptin_2_std = st.stdev(leptin_2)
adiponectin_2_std = st.stdev(adiponectin_2)
resistin_2_std = st.stdev(resistin_2)
mcp1_2_std = st.stdev(mcp1_2)

def show(name, mean1, mean2, std1, std2):
    print(name + ":\nmean 1: " + str(mean1) + "\nmean 2: " + str(mean2) +
          "\nstd 1: " + str(std1) + "\nstd 2: " + str(std2))

show("age", age_1_mean, age_2_mean, age_1_std, age_2_std)
show("bmi", bmi_1_mean, bmi_2_mean, bmi_1_std, bmi_2_std)
show("glucose", glucose_1_mean, glucose_2_mean, glucose_1_std, glucose_2_std)
show("insulin", insulin_1_mean, insulin_2_mean, insulin_1_std, insulin_2_std)
show("homa", homa_1_mean, homa_2_mean, homa_1_std, homa_2_std)
show("leptin", leptin_1_mean, leptin_2_mean, leptin_1_std, leptin_2_std)
show("adiponectin", adiponectin_1_mean, adiponectin_2_mean, adiponectin_1_std, adiponectin_2_std)
show("resistin", resistin_1_mean, resistin_2_mean, resistin_1_std, resistin_2_std)
show("mcp1", mcp1_1_mean, mcp1_2_mean, mcp1_1_std, mcp1_2_std)
