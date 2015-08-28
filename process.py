#!/usr/bin/env python


import sys
import numpy as np
import time
import os.path
import matplotlib.pyplot as plt
from scipy import optimize
import pylab as plb
import cProfile
import pstats
from math import sqrt
from optparse import OptionParser

verbose = False

sugar_atom_nums = {}

sugar_atoms = ["C1", "O1", "HO1", "C2", "O2", "HO2",\
               "C3", "O3", "HO3", "C4", "O4", "HO4",\
               "C5", "O5", "C6",  "O6", "HO6"]

#atomic charges
atomic_charges = {"C4": 0.232, "O4": -0.642, "HO4": 0.410, "C3": 0.232, "O3": -0.642, "HO3": 0.410, "C2": 0.232, "O2": -0.642,\
                  "HO2": 0.410, "C6": 0.232, "O6": -0.642, "HO6": 0.410, "C5": 0.376, "O5": -0.480, "C1": 0.232, "O1": -0.538,\
                  "HO1": 0.410, "HW1": 0.41, "HW2": 0.41, "OW": -0.82}
               
#bond lengths we want to know
bond_pairs = [["C1", "O1"], ["O1", "HO1"], ["C1", "C2"],\
              ["C2", "O2"], ["O2", "HO2"], ["C2", "C3"],\
              ["C3", "O3"], ["O3", "HO3"], ["C3", "C4"],\
              ["C4", "O4"], ["O4", "HO4"], ["C4", "C5"],\
              ["C5", "O5"], ["C5", "C6"],\
              ["C6", "O6"], ["O6", "HO6"]]

#bond angles we want to know
bond_triples = [["O1", "C1", "C2"], ["C1", "O1", "HO1"], ["C1", "C2", "O2"], ["C1", "C2", "C3"],\
                ["O2", "C2", "C3"], ["C2", "O2", "HO2"], ["C2", "C3", "O3"], ["C2", "C3", "C4"],\
                ["O3", "C3", "C4"], ["C3", "O3", "HO3"], ["C3", "C4", "O4"], ["C3", "C4", "C5"],\
                ["O4", "C4", "C5"], ["C4", "O4", "HO4"], ["C4", "C5", "C6"], ["C4", "C5", "O5"],\
                ["C6", "O5", "C1"], ["C5", "C6", "HO6"], ["C5", "O5", "C1"],\
                ["O5", "C1", "O1"], ["O5", "C1", "C2"]]

#bond dihedrals we want to know
bond_quads = []

#names of the cg sites
cg_sites = ["C1", "C2", "C3", "C4", "C5", "O5"]

#atom numbers of the cg sites
cg_atom_nums = {}

#bonds between cg sites
cg_bond_pairs = [["C1", "C2"], ["C2", "C3"], ["C3", "C4"], ["C4", "C5"],\
                 ["C5", "O5"], ["O5", "C1"]]

#bond angles between cg sites
cg_bond_triples = [["O5", "C1", "C2"], ["C1", "C2", "C3"], ["C2", "C3", "C4"],\
                   ["C3", "C4", "C5"], ["C4", "C5", "O5"], ["C5", "O5", "C1"]]

#bond dihedrals between cg sites
cg_bond_quads = [["O5", "C1", "C2", "C3"], ["C1", "C2", "C3", "C4"],\
                 ["C2", "C3", "C4", "C5"], ["C3", "C4", "C5", "O5"],\
                 ["C4", "C5", "O5", "C1"], ["C5", "O5", "C1", "C2"]]

class Atom:
    def __init__(self, atom_type, coords, dipole):
        self.atom_type = atom_type
        self.coords = coords
        self.dipole = dipole

    def __repr__(self):
        return "<Atom {0} charge={1}, mass={2} @ {3}, {4}, {5}>".format(self.atom_type, *self.coords, *self.dipole)

class Frame:
    def __init__(self, num, atom_nums=sugar_atom_nums):
        self.num = num
        self.atoms = []
        self.atom_nums = atom_nums
        self.calc_measure = {"length": self.get_bond_lens, "angle": self.get_bond_angles, "dihedral": self.get_bond_dihedrals}

    def __repr__(self):
        return "<Frame {0} containing {1} atoms>\n{2}".format(self.num, len(self.atoms), self.title)
        
    def bond_length(self, num1, num2):
        """
        return Euclidean distance between two atoms
        """
        dist = np.linalg.norm(self.atoms[num1].loc - self.atoms[num2].loc)
        #print("Distance between atoms {0} and {1} is {2}".format(num1, num2, dist))
        return dist

    def bond_length_atoms(self, atom1, atom2):
        """
        return Euclidean distance between two atoms
        """
        dist = sqrt((atom1.loc[0]-atom2.loc[0])**2 +  (atom1.loc[1]-atom2.loc[1])**2 + (atom1.loc[2]-atom2.loc[2])**2)
        #dist = sqrt(sum((atom1.loc-atom2.loc)**2))
        #dist = np.linalg.norm(atom1.loc - atom2.loc)
        #print("Distance between atoms {0} and {1} is {2}".format(num1, num2, dist))
        return dist
    
    def bond_angle(self, num1, num2, num3, num4):
        """
        angle at atom2 formed by bonds: 1-2 and 2-3
        """
        vec1 = (self.atoms[num2].loc - self.atoms[num1].loc)
        vec2 = (self.atoms[num4].loc - self.atoms[num3].loc)
        #vec1 = vec1 / np.linalg.norm(vec1)
        vec1 = vec1 / sqrt(vec1[0]**2 + vec1[1]**2 + vec1[2]**2)
        #vec2 = vec2 / np.linalg.norm(vec2)
        vec2 = vec2 / sqrt(vec2[0]**2 + vec2[1]**2 + vec2[2]**2)
        angle = np.arccos(np.dot(vec1, vec2))
        return 180 - (angle * 180 / np.pi)
    
    def angle_norm_bisect(self, num1, num2, num3):
        """
        return normal vector to plane formed by 3 atoms and their bisecting vector
        """ 
        vec1 = (self.atoms[num2].loc - self.atoms[num1].loc)
        vec2 = (self.atoms[num3].loc - self.atoms[num2].loc)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        normal = np.cross(vec1, vec2)
        bisec = (vec1 + vec2) / 2.
        return polar_coords(normal), polar_coords(bisec)

    def show_atoms(self, start=0, end=-1):
        """
        print coordinates of the atoms numbered start to end
        """
        print(self.title)
        if end == -1:
            end = len(self.atoms)
        for i in xrange(start, end):
            print(self.atoms[i])

    def get_bond_lens(self, request=bond_pairs):
        dists = np.zeros(len(request))
        for i, pair in enumerate(request):
            dists[i] = self.bond_length(self.atom_nums[pair[0]], self.atom_nums[pair[1]])
        return dists

    def get_bond_angles(self, request=bond_triples):
        angles = np.zeros(len(request))
        for i, triple in enumerate(request):
            angles[i] = self.bond_angle(self.atom_nums[triple[0]], self.atom_nums[triple[1]], self.atom_nums[triple[1]], self.atom_nums[triple[2]])
        return angles

    def get_bond_dihedrals(self, request=bond_quads):
        dihedrals = np.zeros(len(request))
        for i, quad in enumerate(request):
            dihedrals[i] = self.bond_angle(self.atom_nums[quad[0]], self.atom_nums[quad[1]], self.atom_nums[quad[2]], self.atom_nums[quad[3]])
        return dihedrals

def calc_measures(frames, req="length", request=bond_quads, export=True):
    print("Calculating bond "+req+"s")
    if export:
        f = open("bond_"+req+"s.csv", "a")
    t_start = time.clock()
    measures = []
    if export:
        measure_names = ["-".join(name) for name in request]
        f.write(",".join(measure_names) + "\n")
    for i, frame in enumerate(frames):
        perc = i * 100. / len(frames)
        if(i%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        measures.append(frame.calc_measure[req](request))
        if export:
            measures_text = [str(num) for num in measures[-1]]
            f.write(",".join(measures_text) + "\n")
    avg = np.mean(measures, axis=0)
    t_end = time.clock()
    if export:
        f.truncate()
        f.close()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return measures, avg


def polar_coords(xyz, axis1=np.array([0,0,0]), axis2=np.array([0,0,0]), mod=True):
    """
    Convert cartesian coordinates to polar, if axes are given it will be reoriented.
    axis points to the north pole (latitude), axis2 points to 0 on equator (longitude)
    if mod, do angles properly within -pi, +pi
    """
    tpi = 2*np.pi
    polar = np.zeros(3)
    xy = xyz[0]**2 + xyz[1]**2
    polar[0] = np.sqrt(xy + xyz[2]**2)
    polar[1] = np.arctan2(np.sqrt(xy), xyz[2]) - axis1[1]
    polar[2] = np.arctan2(xyz[1], xyz[0]) - axis2[2]
    if axis2[1]<0:
        polar[2] = polar[2] + tpi
    if mod:
        polar[1] = polar[1]%(tpi)
        polar[2] = polar[2]%(tpi)
    return polar

def calc_dipoles(cg_frames, frames, export=True, cg_internal_bonds=cg_internal_bonds, sugar_atom_nums=sugar_atom_nums, adjacent=adjacent):
    """
    starting to think this might be impossible
    dipole of a charged fragment is dependent on where you measure it from
    where should it be measured from??
    
    
    should modify this to include OW dipoles
    modify to drop frames I'm not using - optimise
    """
    print("Calculating dipoles")
    t_start = time.clock()
    old_dipoles = False
    if export:
        f = open("dipoles.csv", "a")
    dipoles = []
    for curr_frame, cg_frame in enumerate(cg_frames):
        perc = curr_frame * 100. / len(cg_frames)
        if(curr_frame%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        frame_dipoles = np.zeros((len(cg_sites),3))
        for i, site in enumerate(cg_sites):
            #if site.atom_type == "OW":
                #continue
            dipole = np.zeros(3)
            #print(site.atom_type)
            if old_dipoles:
                #next 4 lines measure dipole from origin - also inefficient method
                for j, bond in enumerate(cg_internal_bonds[site]):
                    atom1 = frames[curr_frame].atoms[sugar_atom_nums[bond[0]]]
                    atom2 = frames[curr_frame].atoms[sugar_atom_nums[bond[1]]]
                    dipole += (atom1.loc - atom2.loc) * (atom1.charge - atom2.charge)
            else:
                #5 lines measure dipole from centre of charge - seems reasonable
                for atom_name in cg_internal_map[site]:
                    atom = frames[curr_frame].atoms[sugar_atom_nums[atom_name]]
                    dipole += atom.loc * atom.charge #first calc from origin
                cg_atom = cg_frame.atoms[cg_atom_nums[site]]
                dipole -= cg_atom.loc * cg_atom.charge #then recentre it
                #adjust this to check if it's just giving me the coords back
                #dipole += cg_atom.loc
            #next 4 lines measure dipole from cg_bead location - ignores charge on C
            #cg_atom = cg_frame.atoms[cg_atom_nums[site]]
            #for atom_name in cg_internal_map[site]:
                #atom = frames[curr_frame].atoms[sugar_atom_nums[atom_name]]
                #dipole += (atom.loc - cg_atom.loc) * atom.charge
            norm, bisec = cg_frame.angle_norm_bisect(cg_atom_nums[adjacent[site][0]], i, cg_atom_nums[adjacent[site][1]])
            frame_dipoles[i] += polar_coords(dipole, norm, bisec)
            #if frame_dipoles[i][0] < 0.1:
                #print("No dipole--why?")
        if export:
            #f.write("Frame_"+str(curr_frame))
            np.savetxt(f, frame_dipoles, delimiter=",")
        dipoles.append(frame_dipoles)
    if export:
        #f.write("Finished frames")
        f.close()
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return dipoles


def print_output(output_all, output, request):
    for name, val in zip(request, output):
        print("{0}: {1:4.3f}".format("-".join(name), val))


def graph_output(output_all, request):
    rearrange = zip(*output_all)
    plt.figure()
    for i, item in enumerate(rearrange):
        plt.subplot(2,3, i+1)
        data = plt.hist(item, bins=100, normed=1)
        def f(x, a, b, c):
            return a * plb.exp(-(x-b)**2.0 / (2*c**2))
        x = [0.5*(data[1][j] + data[1][j+1]) for j in xrange(len(data[1])-1)]
        y = data[0]
        try:
            popt, pcov = optimize.curve_fit(f, x, y)
            x_fit = plb.linspace(x[0], x[-1], 100)
            y_fit = f(x_fit, *popt)
            plt.plot(x_fit, y_fit, lw=4, color="r")
            print(popt, pcov)
        except RuntimeError:
            print("Failed to optimise fit")


def export_props(grofile, xtcfile, energyfile="", export=False, do_dipoles=False, cutoff=0, cm_map=False):
    global verbose
    t_start = time.clock()
    frames, cg_frames = read_xtc_coarse(grofile, xtcfile, keep_atomistic=do_dipoles, cutoff=cutoff, cm_map=cm_map)
    np.set_printoptions(precision=3, suppress=True)
    if cutoff:
        solvent_rdf(cg_frames, export=export)
    if energyfile:
        read_energy(energyfile, export=export)
    cg_all_dists, cg_dists = calc_measures(cg_frames, "length", cg_bond_pairs, export=export)
    cg_all_angles, cg_angles = calc_measures(cg_frames, "angle", cg_bond_triples, export=export)
    cg_all_dihedrals, cg_dihedrals = calc_measures(cg_frames, "dihedral", cg_bond_quads, export=export)
    if do_dipoles:
        cg_dipoles = calc_dipoles(cg_frames, frames, export)
    if verbose:
        print_output(cg_all_dists, cg_dists, cg_bond_pairs)
        print_output(cg_all_angles, cg_angles, cg_bond_triples)
        print_output(cg_all_dihedrals, cg_dihedrals, cg_bond_quads)
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n".format(len(cg_frames), (t_end - t_start)) + "-"*20)
    return len(cg_frames)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input",
                      action="store", type="string", dest="lammpstrj", default="",
                      help="Input file - LAMMPS trajectory", metavar="FILE")
    # parser.add_option("-v", "--verbose",
    #                   action="store_true", dest="verbose", default=False,
    #                   help="Make more verbose")
    (options, args) = parser.parse_args()
    verbose = options.verbose
    if not options.lammpstrj:
        print("Must provide LAMMPS trajectory to run")
        sys.exit(1)
    if verbose:
        cProfile.run("export_props(options.grofile, options.xtcfile, energyfile=options.energyfile, export=options.export, do_dipoles=options.dipoles, cutoff=options.cutoff, cm_map=options.cm_map)", "profile")
        p = pstats.Stats("profile")
        #p.sort_stats('cumulative').print_stats(25)
        p.sort_stats('time').print_stats(25)
    else:
        export_props(options.grofile, options.xtcfile, energyfile=options.energyfile, export=options.export, do_dipoles=options.dipoles, cutoff=options.cutoff, cm_map=options.cm_map)
