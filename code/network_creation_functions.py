import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from os.path import join

semester_start = '2019-10-01'
semester_end = '2020-02-28'

study_map = {
    'Bachelorstudium'                             : 'bachelor',
    'Masterstudium'                               : 'master',
    'Dr.-Studium d.technischen Wissenschaften'    : 'PhD',
    'Bachelorstudium Lehramt Sek (AB)'            : 'bachelor',
    'Erweiterungsstudium Bachelor (Sek. AB)'      : 'bachelor',
    'Besuch einzelner Lehrveranstaltungen'        : 'non-degree programme',
    'Universitätslehrgang'                        : 'non-degree programme',
    'Dr.-Studium der Naturwissenschaften'         : 'PhD',
    'Lehramtsstudium'                             : 'bachelor & master',
    'Humanmedizin'                                : 'bachelor & master',
    'Doktoratsstudium'                            : 'PhD',
    'Masterstudium Lehramt Sek (AB)'              : 'master',
    'Individuelles Masterstudium'                 : 'master',
    'Dr.-Studium d.montanist. Wissenschaften'     : 'PhD',
    'Erweiterungsstudium'                         : 'bachelor & master',
    'Rechtswissenschaften'                        : 'bachelor & master',
    'PhD-Studium (Doctor of Philosophy)'          : 'PhD',
    'Masterstudium Übersetzen'                    : 'master',
    'Individuelles Bachelorstudium'               : 'bachelor',
    'Dr.-Studium der Philosophie'                 : 'PhD',
    'Bachelorst.Transkulturelle Kommunikation'    : 'bachelor',
    'Masterst. Übersetzen u.Dialogdolmetschen'    : 'master',
    'Dr.-Studium d.Sozial- u.Wirtschaftswiss.'    : 'PhD',
    'Erweiterungsstudium Master (Sek. AB)'        : 'master',
    'Erweiterungsstudium gemäß § 54c UG'          : 'bachelor',
    'Dr.-Studium der angew. med. Wissenschaft'    : 'PhD',
    'Dr.-Studium der Bodenkultur'                 : 'PhD',
    'Bühnengestaltung'                            : 'bachelor & master',
    'Pharmazie'                                   : 'bachelor & master',
    'Doctor of Philosophy-Doktoratsstudium'       : 'PhD',
    'Dr.-Studium der medizin. Wissenschaft'       : 'PhD',
    'Individuelles Diplomstudium'                 : 'bachelor & master',
    'Maschinenbau'                                : 'bachelor & master'
}


def get_study_data(study_id, studies, students, lecturers, groups, rooms, 
                   dates):
    study_data = studies[['study_id', 'study_name']]\
        .drop_duplicates()\
        .set_index('study_id')
    print('data for study {} ({})'\
        .format(study_id, study_data.loc[study_id, 'study_name']))

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
    
    # take all rooms that sample groups are taught in
    sample_room_ids = dates[dates['group_id']\
                    .isin(sample_group_ids)]['room_id'].unique()
    print('\tthe groups are taught in {}/{} of the available rooms'\
        .format(len(sample_room_ids), len(rooms['room_id'].unique())))       

    return (sample_student_ids, sample_students, sample_lecture_ids, 
            sample_group_ids, sample_lecturers, sample_lecturer_ids,
            sample_room_ids)


def add_students(G, student_df, studies_df):
    existing_students = set([n for n, data in G.nodes(data=True) if \
                             data['type'] == 'unistudent'])
    new_students = set(student_df['student_id']).difference(existing_students)
    student_ids = list(new_students)
    print('adding {} students'.format(len(student_ids)))
    
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
    existing_students = set([n for n, data in G.nodes(data=True) if \
                             data['type'] == 'unistudent'])
    new_students = set(student_df['student_id']).difference(existing_students)
    student_ids = list(new_students)
    print('adding {} students'.format(len(student_ids)))
    
    for student_id in student_ids:
        G.add_node(student_id)
        nx.set_node_attributes(G, {student_id:{'type':'student'}})
        
def add_lecturers(G, lecturer_df, organisation_df):
    existing_lecturers = set([n for n, data in G.nodes(data=True) if \
                             data['type'] == 'lecturer'])
    new_lecturers = set(lecturer_df['lecturer_id']).difference(existing_lecturers)
    lecturer_ids = list(new_lecturers)
    print('adding {} lecturers'.format(len(lecturer_ids)))
    
    # TODO: map units to camspuses
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
    existing_lecturers = set([n for n, data in G.nodes(data=True) if \
                             data['type'] == 'lecturer'])
    new_lecturers = set(lecturer_df['lecturer_id']).difference(existing_lecturers)
    lecturer_ids = list(new_lecturers)
    print('adding {} lecturers'.format(len(lecturer_ids)))
    
    for lecturer_id in lecturer_ids:
        G.add_node(lecturer_id)
        nx.set_node_attributes(G, {lecturer_id:{'type':'lecturer'}})

        
def link_group_members(G, group1, group2, wd, day, duration, link_type):
    # student <-> student contacts
    edge_keys = []
    for n1 in group1:
        for n2 in group2:
            tmp = [n1, n2]
            tmp.sort()
            n1, n2 = tmp
            key = '{}{}d{}'.format(n1, n2, wd)
            # no self-loops
            if n1 != n2: 
                # if edge hasn't already been counted for this lecture/group
                if key not in edge_keys:
                    # if the students already have a connection on the same
                    # day through a different lecture or group
                    if G.has_edge(n1, n2, key=key):
                        G[n1][n2][key]['duration'] += duration
                    # if the edge is completely new
                    else:
                        G.add_edge(n1, n2, \
                                   link_type = link_type,
                                   day = day,
                                   weekday = wd,
                                   duration = duration,
                                   key = key)
                    edge_keys.append(key)
                    
                    
def get_group_information(group_dates, students, lecturers, rooms, frac):
    
    assert len(group_dates) == 1
    
    group_id = group_dates['group_id'].values[0]
    # figure out for how long [minutes] the lecture / tutorial went on
    duration = group_dates['duration'].values[0] 
    
    students_in_group = students[\
        students['group_id'] == group_id]['student_id'].unique()

    lecturers_in_group = lecturers[\
        lecturers['group_id'] == group_id]['lecturer_id'].unique()
    
    # if we do not allow for overbooking, we remove excess students that
    # surpass the room's capacity, even if 100% occupancy is allowed
    if frac != 'overbooked':
        # figure out which room the group was taught in and how many seats
        # that room has
        room = group_dates['room_id'].values[0]
        seats = rooms[rooms['room_id'] == room]['seats']
        assert len(seats) == 1
        
        if len(seats) == 0 or seats.values[0] != seats.values[0]:
            seats = np.nan
            print('no seat information for room {} found'.format(room))
        else:
            seats = seats.values[0]

        # remove a fraction of students from the lecture rooms
        if seats == seats: # nan-check
            # the number of students to remove is the difference between the 
            # students that signed up for the lecture and the capacity of the 
            # room, calculated as it's total capacity multiplied by an occupancy
            # fraction
            available_seats = int(np.floor((frac * seats)))
        else:
            available_seats = seats

        students_to_remove = max(0, len(students_in_group) - available_seats)
        if students_to_remove > 0:
            print('removing {}/{} students from room with {:1.0f} seats (occupancy {:1.0f}%)'\
             .format(students_to_remove, len(students_in_group),
                     seats, frac * 100))

        students_in_group = np.random.choice(students_in_group, 
            len(students_in_group) - students_to_remove, replace=False)
        
    return students_in_group, lecturers_in_group, duration


def link_group(G, students_in_group, lecturers_in_group, wd, day, duration):
    # student <-> student contacts
    link_group_members(G, students_in_group, students_in_group, wd, day, 
                       duration, 'student_student_group')
    # student <-> lecturer contacts
    link_group_members(G, students_in_group, lecturers_in_group, wd, day, 
                       duration, 'student_lecturer_group')
    # lecturer <-> lecturer contacts
    link_group_members(G, lecturers_in_group, lecturers_in_group, wd, day, 
                       duration, 'lecturer_lecturer_group')
                    

def add_group_contacts(G, students, lecturers, groups, dates, rooms, day, frac):
    wd = get_weekday(day)
    day = str(day)
    
    day_dates = dates[dates['date'] == pd.to_datetime(day)]
    group_ids = set(groups['group_id'])\
                    .intersection(set(dates['group_id']))

    # most of the following complicated logic is due to the fact that group_id
    # is not unique for different dates on the same day. There can be multiple
    # instances of the same group (i.e. same ID) happen on the same day either
    # at the same time or different times (hopefully not a combination). These
    # cases need to be dealt with differently, to ensure no students or 
    # lecturers are cloned
    for group_id in group_ids:
        # identify all dates on a given day associated with a given group id
        group_dates = day_dates[day_dates['group_id'] == group_id]
        
        # group did not take place on the given day
        if len(group_dates) == 0:
            pass
        
        # simple case: one group ID is associated with one date
        elif len(group_dates) == 1:
            students_in_group, lecturers_in_group, duration = \
                get_group_information(group_dates, students, lecturers, rooms, 
                                      frac)
            link_group(G, students_in_group, lecturers_in_group, wd, day, duration)
            
        # multiple dates for a single group but all start at different times:
        # assume that the same students went to all dates and add contact 
        # durations for all dates accordingly
        elif len(group_dates.drop_duplicates(subset=['start_time'])) == \
             len(group_dates):
            
            for start_time in group_dates['start_time']:
                sub_group_dates = group_dates[group_dates['start_time'] == \
                                              start_time]
                students_in_group, lecturers_in_group, duration = \
                get_group_information(sub_group_dates, students, lecturers, \
                                      rooms, frac)
                link_group(G, students_in_group, lecturers_in_group, wd, day, 
                           duration)
            
        # multiple dates for a single group but all start at the same time and
        # in different rooms:
        # split the students and lecturers into sub-groups. This has already 
        # happened, the sub-group IDs are stored in the dates, students and 
        # lecturers data frames as "new_group_id", which is np.nan if there are
        # no sub-groups. We use these splits to distribute the students and
        # lecturers to different rooms and only create contacts within these
        # rooms
        elif (len(group_dates.drop_duplicates(subset=['start_time'])) == 1) and \
             (len(group_dates.drop_duplicates(subset=['room_id'])) == \
              len(group_dates)):
            # make sure every one of the sub-groups as a new ID
            assert len(group_dates) == len(group_dates['new_group_id'].unique())
            for room in group_dates['room_id']:
                sub_group_dates = group_dates[group_dates['room_id'] == \
                                              room].copy()
                sub_group_dates = sub_group_dates\
                    .drop(columns=['group_id'])\
                    .rename(columns={'new_group_id':'group_id'})
                
                students_in_group, lecturers_in_group, duration = \
                get_group_information(sub_group_dates, students, lecturers, \
                                      rooms, frac)
                link_group(G, students_in_group, lecturers_in_group, wd, day, 
                           duration)
        
        # some dates are at the same time, some at different times
        elif (len(group_dates.drop_duplicates(subset=['start_time'])) > 1) and \
             (len(group_dates.drop_duplicates(subset=['start_time'])) < \
              len(group_dates)):
            
            print('Dealing with edge case: group {} on {}'\
                  .format(group_id, day))
            
            # check if some dates are completely contained within other dates
            to_drop = []
            for n, row1 in group_dates.iterrows():
                start_time1 = row1['start_time']
                duration1 = row1['duration']
                end_time1 = row1['end_time']
                
                for m, row2 in group_dates.iterrows():
                    start_time2 = row2['start_time']
                    duration2 = row2['duration']
                    end_time2 = row2['end_time']
                    
                    if (n != m) and (duration1 < duration2) and \
                       (start_time1 >= start_time2) and \
                       (end_time1 <= end_time2):
                        to_drop.append(n)
                        
            to_drop = list(set(to_drop))
            for index in to_drop:    
                group_dates = group_dates.drop(index)
            
            # split into lectures at the same time but in different rooms and
            # lectures that start at different times
            for start_time in group_dates['start_time'].unique():
                tmp_group_dates = group_dates[group_dates['start_time'] == \
                                              start_time].copy()
                
                # groups that occur at the same time
                if len(tmp_group_dates) > 1:
                    # make sure that groups starting at the same time occur in
                    # different rooms
                    assert len(tmp_group_dates['room_id'].unique()) > 1
                    
                    tmp_group_dates = tmp_group_dates\
                        .drop(columns=['group_id'])\
                        .rename(columns={'new_group_id':'group_id'})
                    
                    # iterate over all locations at which the group occurs
                    for room in tmp_group_dates['room_id']:
                        sub_group_dates = tmp_group_dates[\
                            tmp_group_dates['room_id'] == room].copy()
                            
                        students_in_group, lecturers_in_group, duration = \
                        get_group_information(sub_group_dates, students, lecturers, \
                                              rooms, frac)
                        link_group(G, students_in_group, lecturers_in_group, wd,
                                   day, duration)
        
                # groups that occur at a different time
                else:
                    students_in_group, lecturers_in_group, duration = \
                            get_group_information(tmp_group_dates, students, \
                                        lecturers, rooms, frac)
                    link_group(G, students_in_group, lecturers_in_group, wd, 
                                   day, duration)
           
        else:
            print('something happened that I didnt think could happen!')
        
    
def add_unistudent_contacts(G, level):
    print('{:1.1f}% additional contacts between students'.format(level * 100))
    students = [n[0] for n in G.nodes(data=True) if \
                n[1]['type'] == 'unistudent']
    weekdays = {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday',
            5:'Friday', 6:'Saturday', 7:'Sunday'}

    for wd in range(1, len(weekdays) + 1):
        print()
        # all edges between students on a given weekday
        keys = [e[2] for e in G.edges(keys=True, data=True) if \
                (e[3]['weekday'] == wd) and \
                (e[3]['link_type'] in ['student_student_group',\
                                       'student_student_additional'])]
    
        # we assume that the network only contains one week worth of data.
        # therefore, there should only be one date associated with each weekday
        days = set([e[2]['day'] for e in G.edges(data=True) if \
                e[2]['weekday'] == wd])
        assert len(days) == 1
        day = list(days)[0]
        
        # calculate the number of additional links to add as fraction of the
        # existing links
        N_unistudent_edges = len(keys) 
        N_additional_links = round(level * N_unistudent_edges)
        print('\t{}: adding {} additional links between students'\
              .format(weekdays[wd], N_additional_links))

        # add new links to the network between randomly chosen students (i.e.
        # no dependence on campus, study or semester). If we pick the same
        # student twice (as link source and target) or if the link already
        # exists, we redraw nodes. 
        for i in range(N_additional_links):
            s1 = -1
            s2 = -1
            while s1 == s2 or key in keys:
                s1 = np.random.choice(students)
                s2 = np.random.choice(students)
                tmp = [s1, s2]
                tmp.sort()
                s1, s2 = tmp
                key = '{}{}d{}'.format(s1, s2, wd)
    
            # add new link to the network
            G.add_edge(s1, s2, \
                       link_type = 'student_student_additional',
                       day = day,
                       weekday = wd,
                       group = np.nan,
                       key = key)
            keys.append(key)
            
            
def remove_unistudent_contacts(G, level):
    print('remove {:1.1f}% contacts between students'.format(level * 100))
    students = [n[0] for n in G.nodes(data=True) if \
                n[1]['type'] == 'unistudent']
    weekdays = {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday',
            5:'Friday', 6:'Saturday', 7:'Sunday'}
    
    for wd in range(1, len(weekdays) + 1):
        print()
        # all edges between students on a given weekday
        edges = np.asarray([(e[0], e[1], e[2]) for e in \
                G.edges(keys=True, data=True) if \
                (e[3]['weekday'] == wd) and \
                (e[3]['link_type']  == 'student_student_group')])
        
        N_unistudent_edges = len(edges) 
        N_edges_to_remove = round(level * N_unistudent_edges)
        print('\t{}: removing {} links between students'\
              .format(weekdays[wd], N_edges_to_remove))
        
        edges_to_remove_idx = np.random.choice(range(len(edges)),
                                        size=N_edges_to_remove, replace=False)
        edges_to_remove = edges[edges_to_remove_idx]
        # since numpy arrays convert all contents to objects if one of the
        # contained data type is an object (i.e. the key, which is a str), we
        # need to convert the edge IDs into integers before we can remove them
        # from the graph
        edges_to_remove = [[int(e[0]), int(e[1]), e[2]] for e in edges_to_remove]
        
        G.remove_edges_from(edges_to_remove)
            
    
def create_single_day_network(students, lecturers, studies, organisations, 
                              groups, dates, rooms, day, frac=1):
    G = nx.MultiGraph()
    add_students(G, students, studies)
    add_lecturers(G, lecturers, organisations)
    add_group_contacts(G, students, lecturers, groups, dates, rooms, day,
                       frac)
    return G


def create_network(students, lecturers, studies, organisations, groups, dates, 
                   rooms, days, estudents, electurers, edates, frac=1):
    
    G = nx.MultiGraph()
    add_students(G, students, studies)
    add_students(G, estudents, studies)
    add_lecturers(G, lecturers, organisations)
    add_lecturers(G, electurers, organisations)
    
    for day in days:
        add_group_contacts(G, students, lecturers, groups, dates, rooms, day,
                           frac)
    return G


def map_contacts(G, contact_map, N_weekdays=7):

    for wd in range(1, N_weekdays + 1):
        for n1, n2, day in [(n1, n2, data['day']) \
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


def calculate_duration(row):
    end = row['end_time']
    start = row['start_time']
    if end != end or start != start:
        return np.nan
    
    dummydate = datetime.date.today()
    minutes = (datetime.datetime.combine(dummydate, end) - \
               datetime.datetime.combine(dummydate, start)).seconds / 60
    return minutes


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
    
    
