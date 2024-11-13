import vtk

# 读取VTU文件
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("data/vtk_test/sol_098.vtu")
reader.Update()

# 创建阈值过滤器以提取theta层
threshold_filter = vtk.vtkThreshold()
threshold_filter.SetInputData(reader.GetOutput())
threshold_filter.SetInputArrayToProcess(0, 0, 0, 1, "theta")

# 设置阈值范围
lower_threshold = 0.5  # 设置你的层值下限
upper_threshold = 1.0   # 设置你的层值上限
threshold_filter.SetLowerThreshold(lower_threshold)
threshold_filter.SetUpperThreshold(upper_threshold)
threshold_filter.Update()

# 将UnstructuredGrid转换为PolyData
geometry_filter = vtk.vtkGeometryFilter()
geometry_filter.SetInputConnection(threshold_filter.GetOutputPort())
geometry_filter.Update()

# 创建体素
z_layer = 0.0  # 指定切片的高度
thickness = 10  # 指定体素的厚度

# 创建点和面
points = vtk.vtkPoints()
faces = vtk.vtkCellArray()

poly_data = geometry_filter.GetOutput()
num_points = poly_data.GetNumberOfPoints()

# 遍历提取的点
for i in range(num_points):
    point = poly_data.GetPoint(i)
    x, y, z = point

    # 生成体素的8个顶点
    p0 = (x, y, z_layer)
    p1 = (x + 1, y, z_layer)
    p2 = (x + 1, y + 1, z_layer)
    p3 = (x, y + 1, z_layer)
    p4 = (x, y, z_layer + thickness)
    p5 = (x + 1, y, z_layer + thickness)
    p6 = (x + 1, y + 1, z_layer + thickness)
    p7 = (x, y + 1, z_layer + thickness)

    base_index = points.GetNumberOfPoints()
    points.InsertNextPoint(p0)
    points.InsertNextPoint(p1)
    points.InsertNextPoint(p2)
    points.InsertNextPoint(p3)
    points.InsertNextPoint(p4)
    points.InsertNextPoint(p5)
    points.InsertNextPoint(p6)
    points.InsertNextPoint(p7)

    # 创建面（每个立方体的六个面）
    faces.InsertNextCell(4)  # 底面
    faces.InsertCellPoint(base_index)      # p0
    faces.InsertCellPoint(base_index + 1)  # p1
    faces.InsertCellPoint(base_index + 2)  # p2
    faces.InsertCellPoint(base_index + 3)  # p3

    faces.InsertNextCell(4)  # 上面
    faces.InsertCellPoint(base_index + 4)  # p4
    faces.InsertCellPoint(base_index + 5)  # p5
    faces.InsertCellPoint(base_index + 6)  # p6
    faces.InsertCellPoint(base_index + 7)  # p7

    # 侧面
    for k in range(4):
        faces.InsertNextCell(4)
        faces.InsertCellPoint(base_index + k)
        faces.InsertCellPoint(base_index + ((k + 1) % 4))
        faces.InsertCellPoint(base_index + ((k + 1) % 4) + 4)
        faces.InsertCellPoint(base_index + k + 4)

# 创建多边形数据
poly_data_with_thickness = vtk.vtkPolyData()
poly_data_with_thickness.SetPoints(points)
poly_data_with_thickness.SetPolys(faces)

# 写入新的STL文件
stl_writer = vtk.vtkSTLWriter()
stl_writer.SetFileName("sol_098.stl")
stl_writer.SetInputData(poly_data_with_thickness)
stl_writer.Write()
