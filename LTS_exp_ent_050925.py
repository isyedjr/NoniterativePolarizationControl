from mcculw import ul
from mcculw.enums import ULRange
from mcculw.ul import ULError
from mcculw.device_info import DaqDeviceInfo
import pyvisa as visa
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as ln

#import ID900
#import Connect_2_Instruments as c2i

board_num = 3
channel = 0
ao_range = ULRange.BIP10VOLTS

rm = visa.ResourceManager()
psy = rm.open_resource('GPIB1::11::INSTR')
pol = rm.open_resource('TCPIP::100.65.64.201::inst0::INSTR')

print("Wavelength")
print(pol.query("POL:WAVE?"))

#id900 = c2i.connect_2_ID900()

#id900.plot_Raw_Histograms(10,timeFlag=False)

a0 = 1.284
a1 = 1.2369
a2 = 1.3215
a3 = 1.1308

# config_first_detected_device(0, dev_id_list)
daq_dev_info = DaqDeviceInfo(board_num)

def live_plotter(y1_data, line1, ts):
    #line1,2,3,4 = counts from channel 1,2,3,4
    #lineV = voltage levels applied to sweeping piezoelectric
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
    
        #plt.title('Time Series data')
        f = plt.figure(figsize=(8,6))
        f.set_tight_layout(f)
        #f, (ax,ax2) = plt.subplots(3, 1,figsize=(15,8),sharex=True)
        ax = f.add_subplot(111)
        #ax.set_yscale('log')
        # create a variable for the line so we can later update it
        line1, = ax.plot(y1_data,'lime',alpha=0.8,label="Dot Product")
        ax.set_ylabel('Dot Product')
        #ax.set_yscale("log")
        ax.legend()
        ax.set_ylim([-1,1])

        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y1,y2-data
    line1.set_ydata(y1_data)
    plt.pause(ts)
    
    # return line so we can update it again in the next iteration
    return line1



ao_info = daq_dev_info.get_ao_info()

# calibration
rs1 = input("New Calibration (y/n)?")


if rs1 == "y":
    print("calib")
    r = 20
    tc = 550
    steps = 1024 # 256*2 # pow(2, 16) # (r / steps) * x - 10

    def DACwrite(v0, v1, v2, v3, board_num, r):
        ul.v_out(board_num, 0, ao_range, v0)
        ul.v_out(board_num, 1, ao_range, v1)
        ul.v_out(board_num, 2, ao_range, v2)
        ul.v_out(board_num, 3, ao_range, v3)

    # config_first_detected_device(0, dev_id_list)
    daq_dev_info = DaqDeviceInfo(board_num)

    ao_info = daq_dev_info.get_ao_info()

    DACwrite(0, 0, 0, 0, 3, ao_range)

    v1s1 = []
    v1s2 = []
    v1s3 = []

    for x in range(int(steps)):
        DACwrite(4 * x / steps, 0, 0, 0, 3, ao_range)
        time.sleep(1/10000)
        grab = pol.query(":POLarimeter:SOP?\n")
        pol_meas = grab.split(',')
        v1s1.append(float(pol_meas[1]) / float(pol_meas[0]))
        v1s2.append(float(pol_meas[2]) / float(pol_meas[0]))
        v1s3.append(float(pol_meas[3]) / float(pol_meas[0]))
        pol.clear()

    max_v1_id = 0
    md = 0

    for a in range(len(v1s1) - 1):
        d = math.pow(v1s1[a + 1] - v1s1[0], 2) + math.pow(v1s2[a + 1] - v1s2[0], 2) + math.pow(v1s3[a + 1] - v1s3[0], 2) 
        if d > md:
            md = d
            max_v1_id = a + 1

    v2s1 = []
    v2s2 = []
    v2s3 = []

    for x in range(int(steps)):
        DACwrite(0, 4 * x / steps, 0, 0, 3, ao_range)
        time.sleep(1/10000)
        grab = pol.query(":POLarimeter:SOP?\n")
        pol_meas = grab.split(',')
        v2s1.append(float(pol_meas[1]) / float(pol_meas[0]))
        v2s2.append(float(pol_meas[2]) / float(pol_meas[0]))
        v2s3.append(float(pol_meas[3]) / float(pol_meas[0]))
        pol.clear()

    max_v2_id = 0
    md = 0

    for a in range(len(v2s1) - 1):
        d = math.pow(v2s1[a + 1] - v2s1[0], 2) + math.pow(v2s2[a + 1] - v2s2[0], 2) + math.pow(v2s3[a + 1] - v2s3[0], 2) 
        if d > md:
            md = d
            max_v2_id = a + 1

    # compute vectors
    vs1 = np.matrix([[(v1s1[max_v1_id] + v1s1[0]) / 2], [(v1s2[max_v1_id] + v1s2[0]) / 2], [(v1s3[max_v1_id] + v1s3[0]) / 2]])
    vs2 = np.matrix([[(v2s1[max_v2_id] + v2s1[0]) / 2], [(v2s2[max_v2_id] + v2s2[0]) / 2], [(v2s3[max_v1_id] + v2s3[0]) / 2]])
    vs1 = vs1 / ln.norm(vs1)
    vs2 = vs2 / ln.norm(vs2)

    fg = plt.figure()
    ax = plt.axes(projection = "3d")

    x = []
    y = []
    z = []
    id = []
    for a in range(tc):
        x.append(v1s1[a])
        y.append(v1s2[a])
        z.append(v1s3[a])
        id.append(a)

    ax.scatter3D(x, y, z,c=id)
    ax.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True, color='black')
    ax.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True, color='black')
    ax.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True, color='black')
    ax.quiver(0, 0, 0, vs1[0], vs1[1], vs1[2], length=1, normalize=True, color='g')
    ax.set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1))
    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$S_2$")
    ax.set_zlabel("$S_3$")
    plt.show()
    res = "n"
    res = input("Negate? (y/n)")
    if res == "y":
        vs1 = vs1 * -1

    fg = plt.figure()
    ax = plt.axes(projection = "3d")
    x = []
    y = []
    z = []
    id = []
    for a in range(tc):
        x.append(v2s1[a])
        y.append(v2s2[a])
        z.append(v2s3[a])
        id.append(a)

    ax.scatter3D(x, y, z,c=id)
    ax.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True, color='black')
    ax.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True, color='black')
    ax.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True, color='black')
    ax.quiver(0, 0, 0, vs2[0], vs2[1], vs2[2], length=1, normalize=True, color='g')
    ax.set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1))
    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$S_2$")
    ax.set_zlabel("$S_3$")
    plt.show()

    res = input("Negate? (y/n)")
    if res == "y":
        vs2 = vs2 * -1

    vs3 = np.matrix_transpose(np.cross(np.matrix_transpose(vs1), np.matrix_transpose(vs2)))

    A = np.matrix([[float(vs1[0]), float(vs2[0]), float(vs3[0])], [float(vs1[1]), float(vs2[1]), float(vs3[1])], [float(vs1[2]), float(vs2[2]), float(vs3[2])]])
    B = ln.inv(A)
    print(B)
    np.save('matrix.npy', B)

else:
    B = np.load('matrix.npy')

def DACwrite(dd0, dd1, dd2, dd3, board_num, r):
    v0 = math.pow((dd0 / a0), 0.5)
    v1 = math.pow((dd1 / a1), 0.5)
    v2 = math.pow((dd2 / a2), 0.5)
    v3 = math.pow((dd3 / a3), 0.5)
    if v0 > 4:
        v0 = 0
    if v1 > 4:
        v1 = 0
    if v2 > 4:
        v2 = 0
    if v3 > 4:
        v3 = 0

    ul.v_out(board_num, 0, ao_range, v0)
    ul.v_out(board_num, 1, ao_range, v1)
    ul.v_out(board_num, 2, ao_range, v2)
    ul.v_out(board_num, 3, ao_range, v3)

# measured frame
fi = (np.matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
# set to desired output frame

time.sleep(0.1)
DACwrite(0,0,0,0, 3, ao_range)

DACwrite(0, 0, 0, 0, 3, ao_range)
blank = input("Send in A1 (Enter when done)")
time.sleep(.1)
grab = pol.query(":POLarimeter:SOP?\n")

pol_meas = grab.split(',')
out = np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))]))
plot_a1 = out
fi[0,0] = float(out[0])
fi[1,0] = float(out[1])
fi[2,0] = float(out[2])
ot1 = out
psy.clear()
pol.clear()



blank = input("Send in A2 (Enter when done)")
time.sleep(.1)
grab = pol.query(":POLarimeter:SOP?\n")
pol_meas = grab.split(',')
out = np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))]))
plot_a2 = out

line1 = []
size = 400 
ag = np.zeros(size)

try:
    while True:
        time.sleep(.1)
        grab = pol.query(":POLarimeter:SOP?\n")
        pol_meas = grab.split(',')
        aa2 = np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))]))

        print("Anlge (deg): " + str((180 / np.pi)*np.acos(ot1[0]*aa2[0] + ot1[1]*aa2[1] + ot1[2]*aa2[2])))
        ag[-1] = (ot1[0]*aa2[0] + ot1[1]*aa2[1] + ot1[2]*aa2[2])
        line1 = live_plotter(ag, line1, ts = 0.2)
        ag = np.append(ag[1:],0.0)
except KeyboardInterrupt:
    pass

out = aa2

ot2 = out
fi[0,1] = float(out[0])
fi[1,1] = float(out[1])
fi[2,1] = float(out[2])
pol.clear()
psy.clear()

print(ot1)
print(np.matrix_transpose(ot2))

out = np.matrix_transpose(np.cross(np.matrix_transpose(ot1), np.matrix_transpose(ot2)))

fi[0,2] = float(out[0])
fi[1,2] = float(out[1])
fi[2,2] = float(out[2])

pol.clear()
psy.clear()

# s + pi/2 = c
for k in range(1):
    time.sleep(.25)
    fo = (np.matrix([[math.cos((np.pi/2)*(float(k) / 100)), math.cos((np.pi / 2) + (np.pi/2)*(float(k) / 100)), 0], [math.sin((np.pi/2)*(float(k) / 100)), math.sin((np.pi / 2) + (np.pi/2)*(float(k) / 100)),0],[0,0,1]]))
    
    cheese_touch = input("Connect B1: ")
    grab = pol.query(":POLarimeter:SOP?\n")
    pol_meas = grab.split(',')
    out = np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))]))
    bt1 = out
    fo[0,0] = out[0]
    fo[1,0] = out[1]
    fo[2,0] = out[2]

    cheese_touch = input("Connect B2: ")
    
    line1 = []
    size = 400 
    ag = np.zeros(size)

    try:
        while True:
            time.sleep(.1)
            grab = pol.query(":POLarimeter:SOP?\n")
            pol_meas = grab.split(',')
            bb2 = np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))]))

            print("Angle (deg): " + str((180 / np.pi)*np.acos(bt1[0]*bb2[0] + bt1[1]*bb2[1] + bt1[2]*bb2[2])))
            ag[-1] = (bt1[0]*bb2[0] + bt1[1]*bb2[1] + bt1[2]*bb2[2])
            line1 = live_plotter(ag, line1, ts = 0.2)
            ag = np.append(ag[1:],0.0)
    except KeyboardInterrupt:
        pass

    out = bb2

    bt2 = out
    fo[0,1] = out[0]
    fo[1,1] = out[1]
    fo[2,1] = out[2]

    out = np.matrix_transpose(np.cross(np.matrix_transpose(bt1), np.matrix_transpose(bt2)))

    fo[0,2] = float(out[0])
    fo[1,2] = float(out[1])
    fo[2,2] = float(out[2])

 
    print("Output Fame:\n")
    print(fo)

    Q = ln.inv(np.matmul(np.matmul(B, fo), ln.inv(np.matmul(B,fi))))

    # run algorithm on first frame vector
    xi = float(Q[0,0])
    yi = float(Q[1,0])
    zi = float(Q[2,0])

    # align to Q1
    xd = 1
    yd = 0
    zd = 0

    phi_1 = math.atan2(zi, yi)
    if phi_1 < 0:
        phi_1 = phi_1 + 2*math.pi

    d1 = 3*math.pi / 2 - phi_1
    #d1 = math.atan2(np.sin(d1), np.cos(d1))
    S3n = -1*math.pow(math.pow(zi,2) + math.pow(yi,2),0.5)
    if d1 < 0:
        d1 = d1 + 2*np.pi
    if ((d1 > np.pi)):
        p1 = np.atan2(np.sin(phi_1), np.cos(phi_1))
        d1 = (math.pi/2) - p1
        S3n = -1 * S3n
    # delta 2, r = 1

    phi_2 = math.atan2(S3n, xi)

    if phi_2 < 0:
        phi_2 = phi_2 + 2*math.pi

    d2 = math.acos(xd) - phi_2

    if d2 < 0:
        d2 = d2 + 2*math.pi

    # compute new location of frame vector 2
    M1 = np.matrix([[1, 0, 0],[0, np.cos(d1), np.sin(-1*d1)],[0, np.sin(d1), np.cos(d1)]])
    M2 = np.matrix([[np.cos(d2), 0, np.sin(-1*d2)],[0, 1, 0],[-1*np.sin(-1*d2), 0, np.cos(d2)]])

    v2 = np.matrix([[float(Q[0,1])],[float(Q[1,1])],[float(Q[2,1])]])
    loc_f2 = np.matmul(M2, np.matmul(M1,v2))

    p_3 = np.atan2(loc_f2[2], loc_f2[1])
    phi_3 = p_3[0,0]
    if phi_3 < 0:
        phi_3 = phi_3 + 2*np.pi

    d3 = 2*np.pi - phi_3

    #if d3 > math.pi:
    #    d3 = np.pi - phi_3

    print("%f, %f, %f" % (d1, d2, d3))
    d22 = math.atan2(math.sin(d2), math.cos(d2))

    if d22 > 0:
        da3 = d22
        da2 = 0
    else:
        da3 = 0
        da2 = -1*d22
    DACwrite(d1, da2, da3, d3,3,20)

    time.sleep(0.1)

cheese_touch = input("Connect A1: ")
grab = pol.query(":POLarimeter:SOP?\n")
pol_meas = grab.split(',')
a1_verify = np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))]))
print(a1_verify)
print(plot_a1)

time.sleep(0.1)

cheese_touch = input("Connect A2: ")
grab = pol.query(":POLarimeter:SOP?\n")
pol_meas = grab.split(',')
a2_verify = np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))]))
print(a2_verify)
print(plot_a2)

fig = plt.figure()

fg = fig.add_subplot(1,2,1, projection='3d')
fg2 = fig.add_subplot(1,2,2, projection='3d')
fg.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True, color='black')
fg.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True, color='black')
fg.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True, color='black')
fg2.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True, color='black')
fg2.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True, color='black')
fg2.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True, color='black')
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

fg.plot_surface(x, y, z, cmap="Spectral", alpha=0.15)
fg2.plot_surface(x, y, z, cmap="Spectral", alpha=0.15)

fg.quiver(0,0,0, ot1[0], ot1[1], ot1[2], normalize=True, color='lime')
fg.quiver(0,0,0, ot2[0], ot2[1], ot2[2], normalize=True, color='green')

fg.quiver(0,0,0, bt1[0], bt1[1], bt1[2], normalize=True, color='m')
fg.quiver(0,0,0, bt2[0], bt2[1], bt2[2], normalize=True, color='blue')

fg2.quiver(0,0,0, bt1[0], bt1[1], bt1[2], normalize=True, color='m')
fg2.quiver(0,0,0, bt2[0], bt2[1], bt2[2], normalize=True, color='blue')

fg2.quiver(0,0,0, a1_verify[0], a1_verify[1], a1_verify[2], normalize=True, color='lime')
fg2.quiver(0,0,0, a2_verify[0], a2_verify[1], a2_verify[2], normalize=True, color='green')

fig.show()


np.save("ent_alice_init1.npy", ot1)
np.save("ent_alice_init2.npy", ot2)
np.save("ent_bob_1.npy", bt1)
np.save("ent_bob_2.npy", bt2)
np.save("ent_alice_ver1.npy", a1_verify)
np.save("ent_alice_ver2.npy", a2_verify)

cheese_touch = input("Connect Entangled source (enter when done)")
# id900.plot_Raw_Histograms(10,timeFlag=False)