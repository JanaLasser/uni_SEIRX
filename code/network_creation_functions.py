import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from os.path import join

semester_start = '2019-10-01'
semester_end = '2020-02-28'


def get_study_data(study_id, studies, students, lecturers, groups):
    study_data = studies[['study_id', 'study_name']]\
        .drop_duplicates()\
        .set_index('study_id')
    print('data for study {} ({})'.format(study_id, study_data.loc[study_id, 'study_name']))

    # take all students from the sample study
    sample_student_ids = studies[studies['study_id'] == study_id]\
        ['student_id'].unique()

    # remove all the students that are not in the student data
    sample_student_ids = list(set(sample_student_ids)\
                              .intersection(students['student_id']))
    sample_students = students[students['student_id'].isin(sample_student_ids)]
    print('\tthe study has {}/{} students'\
          .format(len(sample_student_ids), len(students['student_id'].unique())))

    # take all lectures in which the sample students participate
    sample_lecture_ids = sample_students['lecture_id'].unique()
    print('\tthe students participate in {}/{} available lectures'\
        .format(len(sample_lecture_ids), len(students['lecture_id'].unique())))

    # take all groups that are part of the sample lectures
    sample_group_ids = sample_students['group_id'].unique()
    print('\tthe lectures have {} groups and the sample students participate in {} of them'\
        .format(len(groups[groups['lecture_id'].isin(sample_lecture_ids)]),
                len(sample_group_ids)))

    # take all lecturers which are teaching the sample groups
    sample_lecturers = lecturers[lecturers['group_id'].isin(sample_group_ids)]
    sample_lecturer_ids = sample_lecturers['lecturer_id'].unique()
    print('\tthe groups are taught by {}/{} of the available lecturers'\
        .format(len(sample_lecturer_ids), len(lecturers['lecturer_id'].unique())))
    
    return (sample_student_ids, sample_students, sample_lecture_ids, 
            sample_group_ids, sample_lecturers, sample_lecturer_ids)


def add_students(G, student_df, studies_df):
    student_ids = student_df['student_id'].unique()
    # TODO: map units to campuses
    for student_id in student_ids:
        G.add_node(student_id)
        studies = studies_df[studies_df['student_id'] == student_id]\
            .set_index('study_id')
        nx.set_node_attributes(G, {student_id:{
            'type':'unistudent',
            'studies':list(studies.index),
            'terms':{study:studies.loc[study, 'term_number'] for \
                     study in studies.index},
            'unit':1}
        })
        
def add_students_dummy(G, student_df):
    student_ids = student_df['student_id'].unique()
    
    for student_id in student_ids:
        G.add_node(student_id)
        nx.set_node_attributes(G, {student_id:{'type':'student'}})
        
def add_lecturers(G, lecturer_df, organisation_df):
    # TODO: map units to campuses
    lecturer_ids = lecturer_df['lecturer_id'].unique()
    
    for lecturer_id in lecturer_ids:
        G.add_node(lecturer_id)
        orgs = organisation_df[organisation_df['lecturer_id'] == lecturer_id]\
            .set_index('lecturer_id')
        nx.set_node_attributes(G, {lecturer_id:{
            'type':'lecturer',
            'organisations':list(orgs.index),
            'unit':1}
        })
        
def add_lecturers_dummy(G, lecturer_df):
    lecturer_ids = lecturer_df['lecturer_id'].unique()
    
    for lecturer_id in lecturer_ids:
        G.add_node(lecturer_id)
        nx.set_node_attributes(G, {lecturer_id:{'type':'lecturer'}})
        

def add_student_student_group_contacts(G, student_df, group_ids, day):
    wd = get_weekday(day)
    day = str(day)
    for group_id in group_ids:
        students_in_group = student_df[\
            student_df['group_id'] == group_id]['student_id']
        assert len(students_in_group) == len(set(students_in_group))
        
        #print('group {}: {} students'.format(group_id, len(students_in_group)))
        edge_keys = []
        for s1 in students_in_group:
            for s2 in students_in_group:
                tmp = [s1, s2]
                tmp.sort()
                s1, s2 = tmp
                key = '{}{}d{}'.format(s1, s2, wd)
                if s1 != s2 and key not in edge_keys:
                    G.add_edge(s1, s2, \
                               link_type = 'student_student_group',
                               day = day,
                               weekday = wd,
                               group = group_id,
                               key = key)
                    edge_keys.append(key)
        #print('added {} edges for group {}'.format(j, group_id))
        #print()
    #print('added a total of {} edges'.format(i))


    
def add_student_lecturer_group_contacts(G, student_df, lecturer_df, 
                                        group_ids, day):
    wd = get_weekday(day)
    day = str(day)
    for group_id in group_ids:
        students_in_group = student_df[\
            student_df['group_id'] == group_id]['student_id']
        assert len(students_in_group) == len(set(students_in_group))
        lecturers_in_group = lecturer_df[\
            lecturer_df['group_id'] == group_id]['lecturer_id']
        assert len(lecturers_in_group) == len(set(lecturers_in_group))
        
        #print('group {}: {} students, {} lecturers'\
        #      .format(group_id, len(students_in_group), len(lecturers_in_group)))
        edge_keys = []
        for n1 in students_in_group:
            for n2 in lecturers_in_group:
                tmp = [n1, n2]
                tmp.sort()
                n1, n2 = tmp
                key = '{}{}d{}'.format(n1, n2, wd)
                if key not in edge_keys:
                    G.add_edge(n1, n2, \
                               link_type = 'student_lecturer_group',
                               day = day,
                               weekday = wd,
                               group = group_id,
                               key = key)
                    edge_keys.append(key)
        #print('added {} edges for group {}'.format(j, group_id))
        #print()
    #print('added a total of {} edges'.format(i))


def add_group_contacts_half(G, student_df, lecturer_df, group_ids, day, frac):
    wd = get_weekday(day)
    day = str(day)
    for group_id in group_ids:
        students_in_group = student_df[\
            student_df['group_id'] == group_id]['student_id'].unique()

        lecturers_in_group = lecturer_df[\
            lecturer_df['group_id'] == group_id]['lecturer_id'].unique()

        # remove a fraction of students from the lecture rooms
        students_in_group = np.random.choice(students_in_group, 
            int(len(students_in_group) * frac), replace=False)
        
        edge_keys = []
        for s1 in students_in_group:
            for s2 in students_in_group:
                tmp = [s1, s2]
                tmp.sort()
                s1, s2 = tmp
                key = '{}{}d{}'.format(s1, s2, wd)
                if s1 != s2 and key not in edge_keys:
                    G.add_edge(s1, s2, \
                               link_type = 'student_student_group',
                               day = day,
                               weekday = wd,
                               group = group_id,
                               key = key)
                    edge_keys.append(key)

        edge_keys = []
        for n1 in students_in_group:
            for n2 in lecturers_in_group:
                tmp = [n1, n2]
                tmp.sort()
                n1, n2 = tmp
                key = '{}{}d{}'.format(n1, n2, wd)
                if key not in edge_keys:
                    G.add_edge(n1, n2, \
                               link_type = 'student_lecturer_group',
                               day = day,
                               weekday = wd,
                               group = group_id,
                               key = key)
                    edge_keys.append(key)

    
    
def create_single_day_network(students, lecturers, studies, organisations, 
                              dates, day):
    sample_dates = dates[dates['date'] == pd.to_datetime(day)]
    all_groups = students['group_id'].unique()
    groups = set(all_groups).intersection(set(sample_dates['group_id']))
    
    G = nx.MultiGraph()
    add_students(G, students, studies)
    add_lecturers(G, lecturers, organisations)
    add_student_student_group_contacts(G, students, groups, day.date())
    add_student_lecturer_group_contacts(G, students, lecturers, groups,
                                        day.date())
    
    return G


def create_network(students, lecturers, studies, organisations, 
                              dates, days, half=False, frac=1):
    
    all_groups = students['group_id'].unique()
    G = nx.MultiGraph()
    
    for day in days:
        sample_dates = dates[dates['date'] == pd.to_datetime(day)]
        groups = set(all_groups).intersection(set(sample_dates['group_id']))

        add_students(G, students, studies)
        add_lecturers(G, lecturers, organisations)

        if half:
            add_group_contacts_half(G, students, lecturers, groups, day, frac=1)
        else:
            add_student_student_group_contacts(G, students, groups, day.date())
            add_student_lecturer_group_contacts(G, students, lecturers, groups,
                                                day.date())
    
    return G


def map_contacts(G, contact_map, N_weekdays=7):

    for wd in range(1, N_weekdays + 1):
        for n1, n2, group_id, day in [(n1, n2, data['group'], data['day']) \
            for (n1, n2, data) in G.edges(data=True) if data['weekday'] == wd]:

            tmp = [n1, n2]
            tmp.sort()
            n1, n2 = tmp
            key = '{}{}d{}'.format(n1, n2, wd)
            link_type = G[n1][n2][key]['link_type']
            G[n1][n2][key]['contact_type'] = contact_map[link_type]


def get_weekday(date):
    tmp = pd.to_datetime(date)
    wd = datetime.datetime(tmp.year, tmp.month, tmp.day).weekday()
    return wd + 1



def draw_uni_network(G, students, lecturers, day, study_id, dst):
    weekdays = {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday',
            5:'Friday', 6:'Saturday', 7:'Sunday'}
    
    H = nx.MultiGraph()
    add_students_dummy(H, students)
    pos_students = nx.spring_layout(H, k=0.1, seed=42)
    H = nx.MultiGraph()
    add_lecturers_dummy(H, lecturers)
    pos_lecturers = nx.spring_layout(H, k=0.1, seed=42)
    pos_students = nx.drawing.layout.rescale_layout_dict(pos_students, 2)

    pos = pos_students
    pos.update(pos_lecturers)
    
    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_nodes(G, pos_students, ax=ax, node_size=20, alpha=0.5,
                           nodelist=students['student_id'].unique(), 
                           node_color='b')
    nx.draw_networkx_nodes(G, pos_lecturers, ax=ax, node_size=20, alpha=0.5,
                           nodelist=lecturers['lecturer_id'].unique(), 
                           node_color='r')
    
    student_edges = [(e[0], e[1]) for e in G.edges(data=True, keys=True) \
                     if e[3]['link_type'] == 'student_student_group']
    student_lecturer_edges = [(e[0], e[1]) for e in G.edges(data=True, keys=True) \
                     if e[3]['link_type'] == 'student_lecturer_group']
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edgelist=student_edges, 
                           edge_color='b')
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edgelist=student_lecturer_edges, 
                           edge_color='purple')
    wd = get_weekday(day)
    print(weekdays[wd])
    ax.set_title(weekdays[wd], fontsize=16)
    plt.savefig(join(dst, '{}_{}.svg'.format(study_id.replace(' ', '_'),
                                str(pd.to_datetime(day).date()))))
    plt.clf();
    
    
