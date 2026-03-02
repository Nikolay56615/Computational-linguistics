from ontology_repository import OntologyRepository

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "qy?u6E89H2"

repo = OntologyRepository(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

repo.run_custom_query("MATCH (n) DETACH DELETE n")

# 1) Корневой класс
cls_person = repo.create_class("Person", "Человек")
print("Создан корневой класс:", cls_person)

# 2) Подклассы Person
cls_employee = repo.create_class("Employee", "Работник компании", parent_uri=cls_person["uri"])
print("Создан подкласс Person:", cls_employee)

cls_employer = repo.create_class("Employer", "Работодатель компании", parent_uri=cls_person["uri"])
print("Создан подкласс Person:", cls_employer)

# 3) Атрибуты класса Employee
dp_employee_age = repo.add_class_attribute(cls_employee["uri"], "age")
print("Добавлен DatatypeProperty к Employee:", dp_employee_age)

op_employee_works_for = repo.add_class_object_attribute(
    cls_employee["uri"], "worksFor", range_class_uri=cls_employer["uri"]
)
print("Добавлен ObjectProperty к Employee:", op_employee_works_for)

# 4) Объект класса Employee
obj_employee_1 = repo.create_object(cls_employee["uri"], "Челик Чел", "Разработчик")
print("Создан объект Employee:", obj_employee_1)

# 5) Сигнатуры классов
sig_employee = repo.collect_signature(cls_employee["uri"])
print("Сигнатура Employee:", sig_employee)

sig_employer = repo.collect_signature(cls_employer["uri"])
print("Сигнатура Employer:", sig_employer)

# 6) Иерархия классов
employee_parents = repo.get_class_parents(cls_employee["uri"])
employee_children = repo.get_class_children(cls_employee["uri"])
print("Родители Employee:", employee_parents)
print("Потомки Employee:", employee_children)

person_parents = repo.get_class_parents(cls_person["uri"])
person_children = repo.get_class_children(cls_person["uri"])
print("Родители Person:", person_parents)
print("Потомки Person:", person_children)

# 7) Обновление класса
cls_employee_updated = repo.update_class(cls_employee["uri"], "Employee", "Сотрудник компании")
print("Обновлённый класс Employee:", cls_employee_updated)

# 8) Подкласс Employee -> Programmer
cls_programmer = repo.create_class("Programmer", "Программист компании", parent_uri=cls_employee["uri"])
print("Создан подкласс Employee:", cls_programmer)

# 9) Атрибуты класса Programmer
dp_programmer_age = repo.add_class_attribute(cls_programmer["uri"], "age")
print("Добавлен DatatypeProperty к Programmer:", dp_programmer_age)

op_programmer_works_for = repo.add_class_object_attribute(
    cls_programmer["uri"], "worksFor", range_class_uri=cls_employer["uri"]
)
print("Добавлен ObjectProperty к Programmer:", op_programmer_works_for)

# 10) Объект класса Programmer
obj_programmer_1 = repo.create_object(cls_programmer["uri"], "Прогер Прогерович", "Разработчик")
print("Создан объект Programmer:", obj_programmer_1)

# 11) Проверка методов онтологии
ontology_dump = repo.get_ontology()
print("Онтология:", ontology_dump)

root_classes = repo.get_ontology_parent_classes()
print("Корневые классы:", root_classes)

print("Класс Employee:", repo.get_class(cls_employee["uri"]))
print("Объекты класса Employee:", repo.get_class_objects(cls_employee["uri"]))

# 12) Удаляем атрибуты
repo.delete_class_object_attribute(op_programmer_works_for["uri"])
print("ObjectProperty удалён у Programmer")

repo.delete_class_attribute(dp_programmer_age["uri"])
print("DatatypeProperty удалён у Programmer")

# 13) Удаляем объект Employee
repo.delete_object(obj_employee_1["uri"])
print("Объект Employee удалён")
print("Объекты класса Employee:", repo.get_class_objects(cls_employee["uri"]))

# 14) Удаляем класс Employee каскадно (должен удалиться и Programmer с объектами/атрибутами)
repo.delete_class(cls_employee["uri"])
print("Класс Employee удалён (и все остальное)")

repo.close()