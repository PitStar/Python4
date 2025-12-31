#!/usr/bin/env python
# coding: utf-8

# In[37]:


from sqlalchemy import create_engine
import pyedflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# In[38]:


DB_NAME="biosignal_db"
DB_USER="student"
DB_PASSWORD="password"
DB_HOST="localhost"
DB_PORT="5432"


# In[39]:


engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# In[40]:


n_signals = []
signal_labels = []
sigbufs = []
edf_path = "abdominal/r01.edf"


# In[41]:


with pyedflib.EdfReader(edf_path) as edf_reader:
    n_signals = edf_reader.signals_in_file
    signal_labels = [edf_reader.getLabel(i) for i in range(n_signals)]
    sigbufs = [edf_reader.readSignal(i) for i in range(n_signals)]


# In[42]:


df = pd.DataFrame(sigbufs).T
df.columns = signal_labels
sample_rate = int(edf_reader.getSampleFrequency(0))
df['timestamp'] = df.index / sample_rate


# In[43]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[44]:


print("Number of NaN:", df.isna().sum())
df_clean = df.dropna()
print("Number of NaN after cleanup:", df_clean.isna().sum())
print("Sample Rate", sample_rate)


# In[45]:


print("First 5 rows:\n", df.head())
print("Stats by columns:\n", df.describe())


# In[46]:


abdomen_channels = ['Abdomen_1', 'Abdomen_2', 'Abdomen_3', 'Abdomen_4']
abdomen_df = df[abdomen_channels]
abdomen_df.head()


# In[47]:


df['Abdomen_mean'] = abdomen_df.mean(axis=1)
df['Abdomen_mean'].head()


# In[48]:


df_filtered = df[df['Direct_1']> 25]
print(df_filtered.head())


# In[49]:


df['second']=df['timestamp'].astype(int)
df_grouped = df.groupby('second')[abdomen_channels].mean()
df_grouped.head()


# In[50]:


A = df.to_numpy()
mean = np.mean(A, axis=0)
std = np.std(A, axis=0)
print("Mean:", mean)
print("STD:", std)      


# In[51]:


N = -1 + 2*(A-A.min())/(A.max()-A.min())
N


# In[52]:


plt.plot(df['timestamp'], df['Direct_1'], label='Direct_1')
plt.plot(df['timestamp'], df['Abdomen_1'], label='Abdomenal 1')
plt.plot(df['timestamp'], df['Abdomen_2'], label='Abdomenal 2')
plt.plot(df['timestamp'], df['Abdomen_3'], label='Abdomenal 3')
plt.plot(df['timestamp'], df['Abdomen_4'], label='Abdomenal 4')
plt.legend()
plt.grid()
plt.show()


# In[53]:


t = df['timestamp'][:2000].values
signal = df['Abdomen_4'][:2000].values
plt.plot(t, signal, label="Abdomenal 4 (zoomed")
plt.legend()
plt.grid()
plt.show()


# In[54]:


b, a = butter(16, 0.1, btype='low', analog=False)
filtered = filtfilt(b, a, signal)
plt.plot(t, signal, label='Abdomenal 4')
plt.plot(t, filtered, label='Low Frequency harmonics')
plt.legend()
plt.grid()
plt.show()


# In[55]:


h = signal - filtered
plt.plot(t, signal, label='Abdomenal 4 (original)')
plt.plot(t, h, label='High-frequency harmonics ')
plt.legend()
plt.grid()
plt.show()

