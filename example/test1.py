from ontology_repository import OntologyRepository

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "qy?u6E89H2"


def reset_db(repo: OntologyRepository) -> None:
    repo.run_custom_query("MATCH (n) DETACH DELETE n")


def print_snapshot(repo: OntologyRepository, title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    classes = repo.run_custom_query("""
        MATCH (c:Class)
        RETURN c.uri AS uri, c.title AS title
        ORDER BY title
    """)
    print("\n[Classes]")
    for r in classes:
        print(f"- {r['title']}  ({r['uri']})")

    subs = repo.run_custom_query("""
        MATCH (child:Class)-[:SUBCLASS_OF]->(parent:Class)
        RETURN child.title AS child, parent.title AS parent
        ORDER BY parent, child
    """)
    print("\n[SUBCLASS_OF]")
    for r in subs:
        print(f"- {r['child']} -> {r['parent']}")

    dps = repo.run_custom_query("""
        MATCH (dp:DatatypeProperty)-[:DOMAIN]->(c:Class)
        RETURN c.title AS class, dp.title AS dp_title, dp.uri AS dp_uri
        ORDER BY class, dp_title
    """)
    print("\n[DatatypeProperty (DOMAIN)]")
    for r in dps:
        print(f"- {r['class']}: {r['dp_title']}  ({r['dp_uri']})")

    ops = repo.run_custom_query("""
        MATCH (op:ObjectProperty)-[:DOMAIN]->(c:Class)
        MATCH (op)-[:RANGE]->(rc:Class)
        RETURN c.title AS domain, op.title AS op_title, op.uri AS op_uri, rc.title AS range
        ORDER BY domain, op_title
    """)
    print("\n[ObjectProperty (DOMAIN -> RANGE)]")
    for r in ops:
        print(f"- {r['domain']}.{r['op_title']} -> {r['range']}  (type={r['op_uri']})")

    objs = repo.run_custom_query("""
        MATCH (o:Object)-[:`rdf:type`]->(c:Class)
        RETURN o.uri AS uri, o.title AS title, c.title AS class
        ORDER BY class, title
    """)
    print("\n[Objects rdf:type]")
    for r in objs:
        print(f"- {r['class']}: {r['title']}  ({r['uri']})")

    rels = repo.run_custom_query("""
        MATCH (a:Object)-[r]->(b:Object)
        WHERE type(r) <> 'rdf:type'
        RETURN a.title AS from_title, type(r) AS rel_type, b.title AS to_title
        ORDER BY from_title, rel_type, to_title
    """)
    print("\n[Object -> Object relations]")
    if not rels:
        print("- (none)")
    for r in rels:
        print(f"- {r['from_title']} -[{r['rel_type']}]-> {r['to_title']}")

    print("\n[Neo4j Browser queries]")
    print("1) Classes:")
    print("   MATCH (c:Class) RETURN c;")
    print("2) Properties:")
    print("   MATCH (p)-[r:DOMAIN|RANGE]->(c:Class) RETURN p,r,c;")
    print("3) Objects with types:")
    print("   MATCH (o:Object)-[:`rdf:type`]->(c:Class) RETURN o,c;")
    print("4) Object links (excluding rdf:type):")
    print("   MATCH (a:Object)-[r]->(b:Object) WHERE type(r)<>'rdf:type' RETURN a,r,b;")


def test_1_person_company(repo: OntologyRepository) -> None:
    reset_db(repo)

    # Ontology
    cls_person = repo.create_class("Person", "Человек")
    cls_company = repo.create_class("Company", "Компания")

    repo.add_class_attribute(cls_person["uri"], "age")
    op_works_for = repo.add_class_object_attribute(cls_person["uri"], "worksFor", cls_company["uri"])

    # Instances
    obj_company = repo.create_object(cls_company["uri"], "Яндекс", "Компания")
    obj_person = repo.create_object(
        cls_person["uri"],
        "Иван",
        "Сотрудник",
        properties={"age": 28},
        object_properties={"worksFor": obj_company["uri"]}  # ключ по title
    )

    print_snapshot(repo, "TEST 1: Иван работает в Яндексе, возраст 28")


def test_2_book_author(repo: OntologyRepository) -> None:
    reset_db(repo)

    # Ontology
    cls_book = repo.create_class("Book", "Книга")
    cls_author = repo.create_class("Author", "Автор")

    repo.add_class_object_attribute(cls_book["uri"], "writtenBy", cls_author["uri"])

    # Instances
    obj_author = repo.create_object(cls_author["uri"], "Александр Пушкин", "Автор")
    obj_book = repo.create_object(
        cls_book["uri"],
        "Евгений Онегин",
        "Роман",
        object_properties={"writtenBy": obj_author["uri"]}
    )

    print_snapshot(repo, "TEST 2: 'Евгений Онегин' writtenBy Пушкин")


def test_3_inheritance(repo: OntologyRepository) -> None:
    reset_db(repo)

    # Ontology: Person <- Employee <- Programmer
    cls_person = repo.create_class("Person", "Человек")
    cls_employee = repo.create_class("Employee", "Сотрудник", parent_uri=cls_person["uri"])
    cls_programmer = repo.create_class("Programmer", "Программист", parent_uri=cls_employee["uri"])
    cls_company = repo.create_class("Company", "Компания")

    repo.add_class_attribute(cls_person["uri"], "age")
    repo.add_class_object_attribute(cls_employee["uri"], "worksFor", cls_company["uri"])

    sig_prog = repo.collect_signature(cls_programmer["uri"])
    print("\n[Signature Programmer]")
    print(sig_prog)

    # Instances
    obj_company = repo.create_object(cls_company["uri"], "JetBrains", "Компания")
    obj_programmer = repo.create_object(
        cls_programmer["uri"],
        "Пётр",
        "Программист",
        properties={"age": 30},  # поле от Person
        object_properties={"worksFor": obj_company["uri"]}  # связь от Employee
    )

    print_snapshot(repo, "TEST 3: Наследование (Programmer inherits age + worksFor)")


def test_4_multiple_facts(repo: OntologyRepository) -> None:
    reset_db(repo)

    # Ontology
    cls_person = repo.create_class("Person", "Человек")
    cls_city = repo.create_class("City", "Город")
    cls_company = repo.create_class("Company", "Компания")
    cls_country = repo.create_class("Country", "Страна")

    repo.add_class_object_attribute(cls_person["uri"], "livesIn", cls_city["uri"])
    repo.add_class_object_attribute(cls_person["uri"], "worksFor", cls_company["uri"])
    repo.add_class_object_attribute(cls_company["uri"], "locatedIn", cls_country["uri"])

    # Instances
    obj_city = repo.create_object(cls_city["uri"], "Таллин", "Город")
    obj_company = repo.create_object(cls_company["uri"], "Bolt", "Компания")
    obj_country = repo.create_object(cls_country["uri"], "Эстония", "Страна")
    obj_person = repo.create_object(
        cls_person["uri"],
        "Анна",
        "Пользователь",
        object_properties={
            "livesIn": obj_city["uri"],
            "worksFor": obj_company["uri"]
        }
    )

    # Связь компании к стране
    repo.update_object(
        obj_company["uri"],
        obj_company["title"],
        obj_company["description"],
        object_properties={"locatedIn": obj_country["uri"]}
    )

    print_snapshot(repo, "TEST 4: Анна livesIn Таллин, worksFor Bolt, Bolt locatedIn Эстония")


if __name__ == "__main__":
    repo = OntologyRepository(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # test_1_person_company(repo)
    # test_2_book_author(repo)
    # test_3_inheritance(repo)
    test_4_multiple_facts(repo)

    repo.close()