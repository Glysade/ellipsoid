cmd.delete('all')
cmd.load('C:\iona\src\ellipsoid\out.sdf')
tmp = drawEllipsoid([0.85, 0.85, 1.00] , -0.5249992750878543, -0.20172490967013168, 0.42215012548688924, 4.613632259645465, 6.203131118930195, 18.786163190366274, 0.031040816412184757, -0.0348100518264296, -0.9989117718839369, -0.9982645963308905, -0.05111606759410893, -0.02923941425606702, -0.050042616126542136, 0.9980858720198154, -0.03633632680466975)
cmd.load_cgo(tmp, 'ellipsoid-cgo')
cmd.set('cgo_transparency', 0.5, 'ellipsoid-cgo')
