
import vtk

# 读取VTU文件
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("data/vtk/sol_199.vtu")
reader.Update()

# 创建阈值过滤器
threshold_filter = vtk.vtkThreshold()
threshold_filter.SetInputData(reader.GetOutput())
threshold_filter.SetInputArrayToProcess(0, 0, 0, 1, "theta")

# 设置阈值范围
lower_threshold = 0.3  # 设置你的层值下限
upper_threshold = 1.0   # 设置你的层值上限
threshold_filter.SetLowerThreshold(lower_threshold)
threshold_filter.SetUpperThreshold(upper_threshold)
threshold_filter.Update()

# 将UnstructuredGrid转换为PolyData
geometry_filter = vtk.vtkGeometryFilter()
geometry_filter.SetInputConnection(threshold_filter.GetOutputPort())
geometry_filter.Update()

# 扩展高度
transform = vtk.vtkTransform()
transform.Scale(1, 1, 2)  # 设置Z轴扩展因子（2为示例，可根据需要调整）

transform_filter = vtk.vtkTransformFilter()
transform_filter.SetInputConnection(geometry_filter.GetOutputPort())
transform_filter.SetTransform(transform)
transform_filter.Update()

# 写入新的STL文件
stl_writer = vtk.vtkSTLWriter()
stl_writer.SetFileName("output_layer.stl")
stl_writer.SetInputConnection(transform_filter.GetOutputPort())
stl_writer.Write()
