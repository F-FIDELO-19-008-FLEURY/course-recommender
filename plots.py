# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:40:16 2022

@author: luis.pinos-ullauri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
import plotly.offline as pyo

### Function that returns the minimum standards per major
### Returns a 4x3 matrix, where it displays the minimum standard per skill (3 by default)
### and by major (4 by default)
def get_minimum_standards():
    minimum_standards=[]
    #Major 1: average expectance of all skills
    minimum_standards.append([0,0,0])
    #Major 2: larger expectance on skill 1 and lower expectance on skill 3
    minimum_standards.append([0.5,0,-1])
    #Major 3: larger expectance on skill 2 and lower expectance on skill 3
    minimum_standards.append([0,0.5,-1])
    #Major 4: lower expectance on skill 1 and larger expectance on skill 3
    minimum_standards.append([-1,0,0.5])
    minimum_standards=pd.DataFrame(minimum_standards)
    minimum_standards.columns=['exp_prof_sskill1','exp_prof_sskill2','exp_prof_sskill3']
    minimum_standards.index=['major1','major2','major3','major4']
    return minimum_standards

def radar_plot():
    minimum_standards=get_minimum_standards()
    categories=['Expected Profile SSkill1','Expected Profile SSkill2','Expected Profile SSkill3']
    categories = [*categories, categories[0]]
    major1=minimum_standards.iloc[0,:].values.tolist()
    major1 = [*major1, major1[0]]
    major2=minimum_standards.iloc[1,:].values.tolist()
    major2 = [*major2, major2[0]]
    major3=minimum_standards.iloc[2,:].values.tolist()
    major3 = [*major3, major3[0]]
    major4=minimum_standards.iloc[3,:].values.tolist()
    major4 = [*major4, major4[0]]
    
    
    fig = go.Figure(
    data=[
        go.Scatterpolar(r=major1, theta=categories, name='Specialisation 1',textfont_size=20),
        go.Scatterpolar(r=major2, theta=categories, name='Specialisation 2',textfont_size=20),
        go.Scatterpolar(r=major3, theta=categories, name='Specialisation 3',textfont_size=20),
        go.Scatterpolar(r=major4, theta=categories, name='Specialisation 4',textfont_size=20)
    ],
    layout=go.Layout(
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )
    )
    fig.update_layout(legend_font_size=19)
    pyo.plot(fig)
    

def function_comparison_plot():
    #minimum_standards=get_minimum_standards()
    #min_std_test=minimum_standards.iloc[0,0]
    min_std_test=3
    min_skp=1
    max_skp=4
    soft_skill_proficiency = np.linspace(min_skp,max_skp,200)
    skp_left = np.linspace(min_skp,min_std_test,200)
    skp_right = np.linspace(min_std_test,max_skp,200)
    fig,ax=plt.subplots()    
    f1_max=max_skp-min_std_test
    f1_min=min_skp-min_std_test
    #print(f1_max)
    #print(f1_min)
    f1=((soft_skill_proficiency-min_std_test)-f1_min)/(f1_max-f1_min)
    f_value=-f1_min/(f1_max-f1_min)
    f2_left_min=-1*pow(min_skp-min_std_test,2)
    f2_right_max=1*pow(max_skp-min_std_test,2)
    f2_left=(-1*pow(skp_left-min_std_test,2))/(-f2_left_min)*f_value+f_value
    f2_right=(1*pow(skp_right-min_std_test,2))/(f2_right_max)*(1-f_value)+f_value
    f3=(1)/(1+((1-f_value)/f_value)*pow(math.e,3*(min_std_test-soft_skill_proficiency)))
    ff1,=ax.plot(soft_skill_proficiency,f1, 'r',label="Linear")
    ff2r,=ax.plot(skp_right,f2_right, 'y',label="Quadratic")
    ff2l,=ax.plot(skp_left,f2_left, 'y',label="Quadratic")
    ff3,=ax.plot(soft_skill_proficiency,f3, 'g',label="Logistic")
    ax.set_ylim([0,1])
    plt.xlabel("Soft Skill proficiency")
    plt.ylabel("Score")
    plt.axvline(x=min_std_test,ls=':', lw=2,color='b')
    plt.axhline(y=-f1_min/(f1_max-f1_min),ls=':', lw=2,color='b')
    first_legend = ax.legend(handles=[ff1,ff2l,ff3], loc='upper left')
    #first_legend = ax.legend(handles=[ff1,ff3], loc='upper left')
    ax.add_artist(first_legend)
    plt.savefig("function_visual_comparison.png", dpi=300)
    plt.show()
    

#radar_plot()
function_comparison_plot()


"""

r_effect_st=[]
sd_re_student=[0.1,0.2,0.1]
for i in range(len(sd_re_student)):  
    r_effect_st.append(norm.rvs(loc=0,scale=sd_re_student[i],size=10)) 
print(r_effect_st[0])
print(r_effect_st[1])
print(r_effect_st[2])
print(r_effect_st[0][1])
print(r_effect_st[1][2])
"""
