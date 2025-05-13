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

board_num = 3
channel = 0
ao_range = ULRange.BIP10VOLTS

rm = visa.ResourceManager()
psy = rm.open_resource('GPIB1::11::INSTR')
pol = rm.open_resource('TCPIP::100.65.64.201::inst0::INSTR')

a0 = 1.284
a1 = 1.2369
a2 = 1.3215
a3 = 1.1308

print("Wavelength")
print(pol.query("POL:WAVE?"))

# config_first_detected_device(0, dev_id_list)
daq_dev_info = DaqDeviceInfo(board_num)

ao_info = daq_dev_info.get_ao_info()

# calibration
rs1 = input("New Calibration (y/n)?")

if rs1 == "y":
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
        print(x)
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
        print(x)
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

dd1 = []
dd2 = []
dd3 = []

th1 = []
ph1 = []
th2 = []
ph2 = []
th3 = []
ph3 = []

sep = []


it1 = 100

# compute frames
# x = math.cos(2*np.pi*(float(it1) / 100))
# y = math.sin(2*np.pi*(float(it1) / 100))
# z = 0

# x = math.cos((np.pi / 2) + 2*np.pi*(float(it1) / 100))
# y = math.sin((np.pi / 2) + 2*np.pi*(float(it1) / 100))
# z = 0

# x = 0
# y = 0
# z = 1

# write to commands
c1 = ":CONTrol:SOP 1,0,0"
c2 = ":CONTrol:SOP 0,1,0"
c3 = ":CONTrol:SOP 0,0,1"

# measured frame
fi = (np.matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
# set to desired output frame

time.sleep(0.1)
DACwrite(0,0,0,0, 3, ao_range)

DACwrite(0, 0, 0, 0, 3, ao_range)
psy.write(c1)
time.sleep(.1)
grab = pol.query(":POLarimeter:SOP?\n")

pol_meas = grab.split(',')
out = np.matmul(B, np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))])))
fi[0,0] = float(out[0])
fi[1,0] = float(out[1])
fi[2,0] = float(out[2])
psy.clear()
pol.clear()

psy.write(c2)
time.sleep(.1)
grab = pol.query(":POLarimeter:SOP?\n")
pol_meas = grab.split(',')
out = np.matmul(B, np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))])))
fi[0,1] = float(out[0])
fi[1,1] = float(out[1])
fi[2,1] = float(out[2])
pol.clear()
psy.clear()

psy.write(c3)
time.sleep(.1)
grab = pol.query(":POLarimeter:SOP?\n")
pol_meas = grab.split(',')
out = np.matmul(B, np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))])))
fi[0,2] = float(out[0])
fi[1,2] = float(out[1])
fi[2,2] = float(out[2])
pol.clear()
psy.clear()

# s + pi/2 = c
for k in range(it1):
    time.sleep(.25)
    fo = (np.matrix([[math.cos((np.pi/2)*(float(k) / 100)), math.cos((np.pi / 2) + (np.pi/2)*(float(k) / 100)), 0], [math.sin((np.pi/2)*(float(k) / 100)), math.sin((np.pi / 2) + (np.pi/2)*(float(k) / 100)),0],[0,0,1]]))
    psy.clear()

    Q = ln.inv(np.matmul(np.matmul(B, fo), ln.inv(fi)))

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
    dd1.append(d1)
    dd2.append(d2)
    dd3.append(d3)

    # measure output frame 

    psy.write(c1)
    time.sleep(.1)
    grab = pol.query(":POLarimeter:SOP?\n")

    fo1 = [0,0,0]
    fo2 = [0,0,0]
    fo3 = [0,0,0]

    pol_meas = grab.split(',')
    out = (np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))])))
    fo1[0] = float(out[0])
    fo1[1] = float(out[1])
    fo1[2] = float(out[2])
    psy.clear()
    pol.clear()
    th1.append((180 / np.pi) * np.atan2(fo1[1], fo1[0]))
    ph1.append((180 / np.pi) * np.acos(fo1[2] / np.sqrt(math.pow(fo1[0], 2) + math.pow(fo1[1], 2) + math.pow(fo1[2], 2))))
    sep1 = (180 / np.pi) * np.acos(fo1[0] / np.sqrt(math.pow(fo1[0], 2) + math.pow(fo1[1], 2) + math.pow(fo1[2], 2)))


    psy.write(c2)
    time.sleep(.1)
    grab = pol.query(":POLarimeter:SOP?\n")
    pol_meas = grab.split(',')
    out = (np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))])))
    fo2[0] = float(out[0])
    fo2[1] = float(out[1])
    fo2[2] = float(out[2])
    pol.clear()
    psy.clear()
    th2.append((180 / np.pi) * np.atan2(fo2[1], fo2[0]))
    ph2.append((180 / np.pi) * np.acos(fo2[2] / np.sqrt(math.pow(fo2[0], 2) + math.pow(fo2[1], 2) + math.pow(fo2[2], 2))))
    sep2 = (180 / np.pi) * np.acos(fo2[1] / np.sqrt(math.pow(fo2[0], 2) + math.pow(fo2[1], 2) + math.pow(fo2[2], 2)))

    psy.write(c3)
    time.sleep(.1)
    grab = pol.query(":POLarimeter:SOP?\n")
    pol_meas = grab.split(',')
    out =  np.matrix_transpose(np.matrix([(float(pol_meas[1]) / float(pol_meas[0])),(float(pol_meas[2]) / float(pol_meas[0])),(float(pol_meas[3]) / float(pol_meas[0]))]))
    fo3[0] = float(out[0])
    fo3[1] = float(out[1])
    fo3[2] = float(out[2])
    pol.clear()
    psy.clear()
    th3.append((180 / np.pi) * np.atan2(fo3[2], fo3[1]))
    ph3.append((180 / np.pi) * np.acos(fo3[0] / np.sqrt(math.pow(fo3[0], 2) + math.pow(fo3[1], 2) + math.pow(fo3[2], 2))))
    sep3 = (180 / np.pi) * np.acos(fo3[2] / np.sqrt(math.pow(fo3[0], 2) + math.pow(fo3[1], 2) + math.pow(fo3[2], 2)))

    sep.append((sep1 + sep2 + sep3) / 3)

    # plot
    fg.scatter(fi[0,0], fi[1,0], fi[2,0], c='r')
    fg.scatter(fi[0,1], fi[1,1], fi[2,1], c='lime')
    fg.scatter(fi[0,2], fi[1,2], fi[2,2], c='b')
    fg2.scatter(fo1[0], fo1[1], fo1[2], c='r')
    fg2.scatter(fo2[0], fo2[1], fo2[2], c='lime')
    fg2.scatter(fo3[0], fo3[1], fo3[2], c='b')
    
fg.set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1))
fg2.set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1))
fg.set_xlabel("$S_1$")
fg.set_ylabel("$S_2$")
fg.set_zlabel("$S_3$")
fg2.set_xlabel("$S_1$")
fg2.set_ylabel("$S_2$")
fg2.set_zlabel("$S_3$")
fg.set_title("Input Frames (RGB)")
fg2.set_title("Output Frames (RGB)")

fig2 = plt.figure()
fg3 = fig2.add_subplot(1,3,1)
fg32 = fig2.add_subplot(1,3,2)
fg33 = fig2.add_subplot(1,3,3)
fg3.plot(dd1)
fg32.plot(dd2)
fg33.plot(dd3)
fg3.set_xlabel("Index")
fg3.set_ylabel("Retardance (rad)")
fg3.set_title("$\delta_1$ Retardance")

fg32.set_ylabel("Retardance (rad)")
fg32.set_xlabel("Index")
fg32.set_title("$\delta_2$ Retardance")

fg33.set_ylabel("Retardance (rad)")
fg33.set_xlabel("Index")
fg33.set_title("$\delta_3$ Retardance")

fig3 = plt.figure()
ff1 = fig3.add_subplot(1,3,1)
ff2 = fig3.add_subplot(1,3,2)
ff3 = fig3.add_subplot(1,3,3)
ff1.hist2d(th1, ph1, [10,10])
ff2.hist2d(th2, ph2, [10,10])
ff3.hist2d(th3, ph3, [10,10])
ff1.set_xlabel("$\phi$ (deg)")
ff1.set_ylabel("$\Theta$ (deg)")
ff1.set_title("2D Angular Histogram (S1)")
ff2.set_xlabel("$\phi$ (deg)")
ff2.set_ylabel("$\Theta$ (deg)")
ff2.set_title("2D Angular Deviation Histogram (S2)")
ff3.set_xlabel("$\phi$ (deg)")
ff3.set_ylabel("$\Theta$ (deg)")
ff3.set_title("2D Angular Deviation Histogram (S3)")
plt.show()