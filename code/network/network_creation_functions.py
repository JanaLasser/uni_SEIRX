import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from os.path import join

######################################
###                                ### 
###   DATA CLEANING FUNCTIONALITY  ###
###                                ###
######################################

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
    'Maschinenbau'                                : 'bachelor & master',
    'Erweiterungsstudium Bachelor'                : 'bachelor'
}

def calculate_duration(row):
    '''Calculate event duration in minutes given a start and end time'''
    end = row['end_time']
    start = row['start_time']
    if end != end or start != start:
        return np.nan
    
    dummydate = datetime.date.today()
    minutes = (datetime.datetime.combine(dummydate, end) - \
               datetime.datetime.combine(dummydate, start)).seconds / 60
    return minutes


def calculate_end_time(row):
    '''Calculate an end time given a start time and a duration'''
    if row["imputed_end_time"]:
        start = row['start_time']
        duration = row["duration"]
        dummydate = datetime.date.today()
        end = datetime.datetime.combine(dummydate, start) + \
          datetime.timedelta(minutes=int(duration))
        return end.time()
    else:
        return row["end_time"]


def overlap(participations, start_time, end_time):
    '''
    Check whether a new event, represented by its start_time and end_time
    overlaps with events the student already committed to participate in.
    '''
    for _, p in participations.iterrows():
        # start time within another event
        if (start_time >= p["start_time"]) and \
           (start_time <= p["end_time"]):
            #print("conflict detected")
            return True
        # end time within another event
        if (end_time >= p["start_time"]) and \
           (end_time <= p["end_time"]):
            #print("conflict detected")
            return True
        # event completely enclosing other event
        if (start_time <= p["start_time"] and \
            end_time >= p["end_time"]):
            return True
        # event completely enclosed by other event
        if (start_time >= p["start_time"] and \
            end_time <= p["end_time"]):
            return True
    # no overlap detected
    return False


def determine_event_participation(day_events, tmp_event_participations,
                                  student_id, date):
    '''
    Given a list of enrollments to events on a given day, determine which events
    the student will participate in, avoiding overlap between events.
    '''
    if len(day_events) > 0:
        for i, event in day_events.iterrows():
            if not overlap(
                tmp_event_participations,
                event["start_time"],
                event["end_time"]
            ):
                tmp_event_participations = tmp_event_participations.append({
                    "student_id":student_id,
                    "group_id":event["group_id"],
                    "date":date,
                    "start_time":event["start_time"],
                    "end_time":event["end_time"]
                }, ignore_index=True)
            else:
                pass
    return tmp_event_participations


def determine_event_supervision(day_events, tmp_event_supervision, 
                                lecturer_id, date):
    '''
    Given a list of supervisions of events on a given day, determine which 
    events the lecturer will supervise, avoiding overlap between events.
    '''
    if len(day_events) > 0:
        for i, event in day_events.iterrows():
            if not overlap(
                tmp_event_supervision,
                event["start_time"],
                event["end_time"]
            ):
                tmp_event_supervision = tmp_event_supervision.append({
                    "lecturer_id":lecturer_id,
                    "group_id":event["group_id"],
                    "date":date,
                    "start_time":event["start_time"],
                    "end_time":event["end_time"]
                }, ignore_index=True)
            else:
                pass
    return tmp_event_supervision


# deprecated?
'''
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
'''

######################################
###                                ### 
### NETWORK CREATION FUNCTIONALITY ###
###                                ###
######################################

def add_students(G, students, verbose):
    '''
    Adds students as nodes to a graph from a table with student information.
    Student information is added to the nodes as node attributes.

    Parameters:
    -----------
    G : networkx MultiGraph
        Graph object to which the students will be added as nodes.
    students: pandas DataFrame
        Table with the information about the students. Is expected to have 
        the columns "student_id", "maun_study", "study_label", "study_level"
        and "term_number".
    verbose: bool
        Whether or not this function should report stats to stdout.
    '''

    existing_students = set([n for n, data in G.nodes(data=True) if \
                             data['type'] == 'unistudent'])
    new_students = set(students['student_id']).difference(existing_students)
    student_ids = list(new_students)
    students = students[students['student_id'].isin(new_students)]
    students = students.set_index(['student_id', 'study_id'])

    if verbose:
        print('\tadding {} students'.format(len(student_ids)))
        
    no_study_found = 0
    for student_id in student_ids:
        # get the main study and the term number for the main study
        main_study = students.loc[student_id, 'main_study'].values[0]
        study_type = students.loc[student_id, main_study]['study_label']
        study_level = students.loc[student_id, main_study]['study_level']
        term = students.loc[student_id, main_study]['term_number']
        if term != term: # nan-check
            no_study_found += 1
        
        # add the student as a node to the network and all information
        # we have for the student as node attributes.
        # Note: the attribute "unit" is a meaningless artifact that we
        # need to include to satisfy the design conditions of the contact
        # network for the agent based simulation
        G.add_node(student_id)
        nx.set_node_attributes(G, {student_id:{
            'type':'unistudent',
            'main_study':main_study,
            'study_type':study_type,
            'study_level':study_level,
            'term':term,
            'unit':1} 
        })

    if verbose:
        print('\tno study found for {} students'.format(no_study_found))
        
        
def add_students_dummy(G, students):
    '''
    Adds students as nodes to a graph from a table with student information.
    This is a dummy function that is just used by the plotting functionality
    and therefore does not add additional student information to the nodes
    as attributes.

    Parameters:
    -----------
    G : networkx MultiGraph
        Graph object to which the students will be added as nodes.
    students: pandas DataFrame
        Table with the information about the students. Is expected to have 
        the column "student_id".
    '''
    existing_students = set([n for n, data in G.nodes(data=True) if \
                             data['type'] == 'unistudent'])
    new_students = set(students['student_id']).difference(existing_students)
    student_ids = list(new_students)
    print('\tadding {} students'.format(len(student_ids)))
    
    for student_id in student_ids:
        G.add_node(student_id)
        nx.set_node_attributes(G, {student_id:{'type':'unistudent'}})
        
        
def add_lecturers(G, lecturers, verbose):
    existing_lecturers = set([n for n, data in G.nodes(data=True) if \
                             data['type'] == 'lecturer'])
    new_lecturers = set(lecturers['lecturer_id']).difference(existing_lecturers)
    lecturer_ids = list(new_lecturers)

    if verbose:
        print('\tadding {} lecturers'.format(len(lecturer_ids)))
    
    # TODO: map units to camspuses
    for lecturer_id in lecturer_ids:
        G.add_node(lecturer_id)
        orgs = lecturers[lecturers['lecturer_id'] == lecturer_id]#\
            #.sample(1, random_state=42)\
            #["organisation_name"].values[0]
        nx.set_node_attributes(G, {lecturer_id:{
            'type':'lecturer',
            'organisations':list(orgs["organisation_name"].values),
            'unit':1}
        })
        
def add_lecturers_dummy(G, lecturers):
    existing_lecturers = set([n for n, data in G.nodes(data=True) if \
                             data['type'] == 'lecturer'])
    new_lecturers = set(lecturers['lecturer_id']).difference(existing_lecturers)
    lecturer_ids = list(new_lecturers)
    print('\tadding {} lecturers'.format(len(lecturer_ids)))
    
    for lecturer_id in lecturer_ids:
        G.add_node(lecturer_id)
        nx.set_node_attributes(G, {lecturer_id:{'type':'lecturer'}})

        
def link_event_members(G, group1, group2, wd, day, date, duration, event_type,
                       link_type):
    edge_keys = []
    for n1 in group1:
        for n2 in group2:
            tmp = [n1, n2]
            tmp.sort()
            n1, n2 = tmp
            key = '{}{}d{}'.format(n1, n2, day)
            # no self-loops
            if n1 != n2: 
                # if edge hasn't already been counted for this event
                if key not in edge_keys:
                    # if the nodes already have a connection on the same
                    # day through a different event
                    if G.has_edge(n1, n2, key=key):
                        G[n1][n2][key]['duration'] += duration
                    # if the edge is completely new
                    else:
                        G.add_edge(n1, n2, \
                                   link_type = link_type,
                                   event_type = event_type,
                                   date = date,
                                   day = day,
                                   weekday = wd,
                                   duration = duration,
                                   key = key)
                    edge_keys.append(key)
                    
                    
def get_event_information(group_dates, day_participation, day_supervision,
                          rooms, frac, verbose):
    
    assert len(group_dates) == 1
    group_id = group_dates["group_id"].values[0]
    
    # figure out for how long [minutes] the event went on
    duration = group_dates['duration'].values[0] 
    
    # figure out which lecture type the event belongs to
    event_type = group_dates["course_type"].values[0]
    
    students_in_event = day_participation[\
        day_participation["group_id"] == group_id]['student_id'].unique()

    lecturers_in_event = day_supervision[\
        day_supervision["group_id"] == group_id]['lecturer_id'].unique()

    # figure out which room the group was taught in and how many seats
    # that room has
    room_id = group_dates['room_id'].values[0]
    if room_id != room_id: # nan-check
        # for exams: allow all enrolled students
        if event_type == "EX":
            seats = len(students_in_event)
        # for non-exams: use the median seat number of rooms in which
        # courses of the same type are held
        else:
            course_type_room_ids = group_dates[\
                group_dates["course_type"] == event_type]["room_id"]\
                .dropna().values
            seats = rooms[rooms["room_id"]\
                .isin(course_type_room_ids)]["seats"].median()

            # if no seat information is available for the given course type,
            # use the overall median number of seats (only happens for "PR")
            if seats != seats:
                seats = rooms["seats"].median()
    else:
        seats = rooms[rooms['room_id'] == room_id]['seats'].values[0]

    # if we do not allow for overbooking, we remove excess students that
    # surpass the room's capacity, even if 100% occupancy is allowed
    if frac != 'overbooked':
        if seats != seats: # nan-check
            # if no seat information is given, use the median seat number
            # of rooms in which courses of the same type are held
            course_type_room_ids = group_dates[\
                group_dates["course_type"] == event_type]["room_id"]\
                .dropna().values
            seats = rooms[rooms["room_id"]\
                .isin(course_type_room_ids)]["seats"].median()

            # if no seat information is available for the given course type,
            # use the overall median number of seats (only happens for "PR")
            if seats != seats:
                seats = rooms["seats"].median()

        # remove a fraction of students from the lecture rooms
        # the number of students to remove is the difference between the 
        # students that signed up for the lecture and the capacity of the 
        # room, calculated as it's total capacity multiplied by an occupancy
        # fraction
        available_seats = int(np.floor((frac * seats)))

        students_to_remove = max(0, len(students_in_event) - available_seats)
        if students_to_remove > 0:
            if verbose:
                print('removing {}/{} students from room with {:1.0f} seats (occupancy {:1.0f}%)'\
                .format(students_to_remove, len(students_in_event),
                     seats, frac * 100))

        # fix the seed and remove the surplus students
        np.random.seed(42)
        students_in_event = np.random.choice(students_in_event, 
            len(students_in_event) - students_to_remove, replace=False)
        
    return students_in_event, lecturers_in_event, duration, event_type


def link_event(G, students_in_event, lecturers_in_event, wd, day, date,
               duration, event_type):
    # student <-> student contacts
    link_event_members(G, students_in_event, students_in_event, wd, day, 
                       date, duration, event_type, 'student_student')
    # student <-> lecturer contacts
    link_event_members(G, students_in_event, lecturers_in_event, wd, day, 
                       date, duration, event_type, 'student_lecturer')
    # lecturer <-> lecturer contacts
    link_event_members(G, lecturers_in_event, lecturers_in_event, wd, day, 
                       date, duration, event_type, 'lecturer_lecturer')
                    

def add_event_contacts(G, students, lecturers, day_participation, day_supervision, 
                       day_dates, rooms, day, date, frac, verbose):
    wd = get_weekday(day)
    date = str(date)

    group_ids = set(day_dates["group_id"])

    # most of the following complicated logic is due to the fact that group_id
    # is not unique for different dates on the same day. There can be multiple
    # instances of the same group (i.e. same ID) happen on the same day either
    # at the same time or different times (hopefully not a combination). These
    # cases need to be dealt with differently, to ensure no students or 
    # lecturers are cloned
    for group_id in group_ids:
        # identify all dates on a given day associated with a given group ID
        group_dates = day_dates[day_dates["group_id"] == group_id]
        
        # simple case: one group ID is associated with one date
        if len(group_dates) == 1:
            students_in_group, lecturers_in_group, duration, event_type = \
                get_event_information(group_dates, day_participation, \
                                      day_supervision, rooms, frac, verbose)
            link_event(G, students_in_group, lecturers_in_group, wd, day, 
                       date, duration, event_type)

        
            
        # multiple dates for a single event but all start at different times:
        # assume that the same students went to all dates and add contact 
        # durations for all dates accordingly
        # NOTE: de-duplication of groups with the same group_id happening at
        # the same time has already been performed in the data cleaning step.
        elif len(group_dates.drop_duplicates(subset=['start_time'])) == \
             len(group_dates):
            
            for start_time in group_dates['start_time']:
                sub_group_dates = group_dates[group_dates['start_time'] == \
                                              start_time]
                students_in_group, lecturers_in_group, duration, event_type = \
                get_event_information(sub_group_dates, day_participation, \
                                      day_supervision, rooms, frac, verbose)
                link_event(G, students_in_group, lecturers_in_group, wd, day, 
                       date, duration, event_type)

        else:
            print("something happened that I didn't think could happen!")
        

# day information not updated yet     
'''
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
                (e[3]['link_type'] in ['student_student',\
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
            
'''

def create_network(students, lecturers, event_dates, rooms, days,
                   event_participation, event_supervision, frac=1, verbose=False):
    
    G = nx.MultiGraph()

    # add students from lecture data
    add_students(G, students, verbose)
    add_lecturers(G, lecturers, verbose)
    
    for day, date in enumerate(days):
        day_participation = event_participation[\
            event_participation["date"] == pd.to_datetime(date)]
        day_supervision = event_supervision[\
        event_supervision["date"] == pd.to_datetime(date)]
        day_dates = event_dates[event_dates['date'] == pd.to_datetime(date)]

        # add connections between students and lecturers 
        add_event_contacts(G, students, lecturers, day_participation, 
                           day_supervision, day_dates, rooms, day + 1,
                           date, frac, verbose)
    return G


def map_contacts(G, contact_map, N_days=7):

    for d in range(1, N_days + 1):
        for n1, n2, day in [(n1, n2, data['day']) \
            for (n1, n2, data) in G.edges(data=True) if data['day'] == d]:

            tmp = [n1, n2]
            tmp.sort()
            n1, n2 = tmp
            key = '{}{}d{}'.format(n1, n2, d)
            link_type = G[n1][n2][key]['link_type']
            G[n1][n2][key]['contact_type'] = contact_map[link_type]


def get_weekday(date):
    tmp = pd.to_datetime(date)
    wd = datetime.datetime(tmp.year, tmp.month, tmp.day).weekday()
    return wd + 1


###########################################
###                                     ### 
### NETWORK VISUALIZATION FUNCTIONALITY ###
###                                     ###
###########################################


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
    
    
