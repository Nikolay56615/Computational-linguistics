from neo4j_repository import Neo4jRepository

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "qy?u6E89H2"

repo = Neo4jRepository(URI, USER, PASSWORD)

# 1) Создать узлы
n1 = repo.create_node({"title": "Person A", "description": "First person"}, labels=["Person"])
n2 = repo.create_node({"title": "Person B", "description": "Second person"}, labels=["Person"])

print("Created:", n1, n2)

# 2) Создать дугу
arc = repo.create_arc(n1["uri"], n2["uri"], rel_type="KNOWS", props={"since": 2020})
print("Arc:", arc)

# 3) Получить всё с дугами
all_with_arcs = repo.get_all_nodes_and_arcs()
print("All nodes and arcs:", all_with_arcs)

# 4) Обновление
updated = repo.update_node(n1["uri"], {"title": "Person A (updated)", "age": 30})
print("Updated:", updated)

# 5) Поиск по меткам
people = repo.get_nodes_by_labels(["Person"])
print("People:", people)

# 6) Удаление дуги и узла
if arc and arc.get("id"):
    ok = repo.delete_arc_by_id(arc["id"])
    print("Arc deleted:", ok)

# ok = repo.delete_node_by_uri(n1["uri"])
# print("Node deleted:", ok)
# ok = repo.delete_node_by_uri(n2["uri"])
# print("Node deleted:", ok)
print(repo.run_custom_query("MATCH (n) RETURN count(n) AS total"))
# print(repo.run_custom_query(
#     """
#     MATCH (n {uri: $uri})
#     RETURN n.title AS title, n.description AS desc
#     """,
#     uri = "node_abc123",
#     limit = 5
# ))
print(repo.run_custom_query("MATCH (n) DETACH DELETE n"))
repo.close()
