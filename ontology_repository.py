# ontology_repository.py
from typing import List, Dict, Any, Optional
from neo4j_repository import Neo4jRepository, TNode


class OntologyRepository(Neo4jRepository):
    """
    Репозиторий для работы с онтологиями поверх графовой БД Neo4j
    """

    # -----------------------
    # Базовые методы
    # -----------------------
    def get_ontology(self) -> List[Dict[str, Any]]:
        """
        Получить всю онтологию (все классы и их связи).
        """
        cypher = "MATCH (c:Class) OPTIONAL MATCH (c)-[r]->(x) RETURN c, r, x"
        return self.run_custom_query(cypher)

    def get_ontology_parent_classes(self) -> List[Dict[str, Any]]:
        """
        Получить корневые классы (без родителей).
        """
        cypher = """
        MATCH (c:Class)
        WHERE NOT (c)-[:SUBCLASS_OF]->(:Class)
        RETURN c
        """
        return self.run_custom_query(cypher)

    def get_class(self, class_uri: str) -> Optional[TNode]:
        """
        Получить класс по uri.
        """
        cypher = "MATCH (c:Class {uri: $uri}) RETURN c"
        res = self.run_custom_query(cypher, {"uri": class_uri})
        return res[0]["c"] if res else None

    def get_class_parents(self, class_uri: str) -> List[Dict[str, Any]]:
        """
        Получить родителей класса.
        """
        cypher = """
        MATCH (c:Class {uri: $uri})-[:SUBCLASS_OF]->(parent:Class)
        RETURN parent
        """
        return self.run_custom_query(cypher, {"uri": class_uri})

    def get_class_children(self, class_uri: str) -> List[Dict[str, Any]]:
        """
        Получить потомков класса.
        """
        cypher = """
        MATCH (parent:Class {uri: $uri})<-[:SUBCLASS_OF]-(child:Class)
        RETURN child
        """
        return self.run_custom_query(cypher, {"uri": class_uri})

    def get_class_objects(self, class_uri: str) -> List[Dict[str, Any]]:
        """
        Получить объекты данного класса.
        """
        cypher = """
        MATCH (o:Object)-[:`rdf:type`]->(:Class {uri: $uri})
        RETURN o
        """
        return self.run_custom_query(cypher, {"uri": class_uri})

    def update_class(self, class_uri: str, title: str, description: str) -> Optional[TNode]:
        """
        Обновить имя и описание класса.
        """
        cypher = """
        MATCH (c:Class {uri: $uri})
        SET c.title = $title, c.description = $description
        RETURN c
        """
        res = self.run_custom_query(cypher, {"uri": class_uri, "title": title, "description": description})
        return res[0]["c"] if res else None

    def create_class(self, title: str, description: str, parent_uri: Optional[str] = None) -> TNode:
        """
        Создать новый класс, при необходимости указать родителя.
        """
        new_class = self.create_node({"title": title, "description": description}, labels=["Class"])
        if parent_uri:
            self.create_arc(new_class["uri"], parent_uri, "SUBCLASS_OF")
        return new_class

    def delete_class(self, class_uri: str) -> bool:
        """
        Удаляет класс вместе со всеми потомками, их объектами и атрибутами.
        """
        # 1. Находим все классы: целевой и его потомков
        cypher_classes = """
        MATCH (c:Class {uri: $uri})
        OPTIONAL MATCH (c)<-[:SUBCLASS_OF*0..]-(descendant:Class)
        WITH collect(DISTINCT c) + collect(DISTINCT descendant) AS classes_to_delete
        UNWIND classes_to_delete AS cls
        RETURN cls.uri AS uri
        """
        res = self.run_custom_query(cypher_classes, {"uri": class_uri})
        classes_uris = [row["uri"] for row in res if row.get("uri")]
        if not classes_uris:
            return False

        # 2. Находим все объекты этих классов
        cypher_objects = """
        MATCH (o:Object)
        WHERE o.class_uri IN $classes
        OR EXISTS { MATCH (o)-[:`rdf:type`]->(c:Class) WHERE c.uri IN $classes }
        RETURN collect(DISTINCT o.uri) AS objects_to_delete
        """
        res_objects = self.run_custom_query(cypher_objects, {"classes": classes_uris})
        object_uris = res_objects[0].get("objects_to_delete", []) if res_objects else []

        # 3. Находим все DatatypeProperty и ObjectProperty этих классов
        cypher_props = """
        MATCH (p)
        WHERE (p:DatatypeProperty OR p:ObjectProperty)
        AND EXISTS { MATCH (p)-[:DOMAIN]->(c:Class) WHERE c.uri IN $classes }
        RETURN collect(p.uri) AS props_to_delete
        """

        res_props = self.run_custom_query(cypher_props, {"classes": classes_uris})
        prop_uris = res_props[0].get("props_to_delete", []) if res_props else []

        # 4. Удаляем объекты
        for obj_uri in object_uris:
            self.delete_node_by_uri(obj_uri)

        # 5. Удаляем свойства
        for prop_uri in prop_uris:
            self.delete_node_by_uri(prop_uri)

        # 6. Удаляем классы (включая потомков)
        for c_uri in classes_uris:
            self.delete_node_by_uri(c_uri)

        return True

    # -----------------------
    # Атрибуты классов
    # -----------------------
    def add_class_attribute(self, class_uri: str, attr_name: str) -> TNode:
        """
        Добавить DatatypeProperty к классу.
        """
        prop = self.create_node({"title": attr_name}, labels=["DatatypeProperty"])
        self.create_arc(prop["uri"], class_uri, "DOMAIN")
        return prop

    def delete_class_attribute(self, prop_uri: str) -> bool:
        """
        Удалить DatatypeProperty (только если это DatatypeProperty).
        """
        cypher = """
        MATCH (p:DatatypeProperty {uri: $uri})
        DETACH DELETE p
        RETURN COUNT(p) > 0 AS deleted
        """
        res = self.run_custom_query(cypher, {"uri": prop_uri})
        return res[0]["deleted"] if res else False

    def add_class_object_attribute(self, class_uri: str, attr_name: str, range_class_uri: str) -> TNode:
        """
        Добавить ObjectProperty.
        """
        prop = self.create_node({"title": attr_name}, labels=["ObjectProperty"])
        # привязываем к классу
        self.create_arc(prop["uri"], class_uri, "DOMAIN")
        # задаём range (с каким классом связан)
        self.create_arc(prop["uri"], range_class_uri, "RANGE")
        return prop

    def delete_class_object_attribute(self, object_property_uri: str) -> bool:
        """
        Удалить ObjectProperty и все связи объектов, которые используют его uri как тип отношения.
        """
        # находим тип отношения (это же uri у ObjectProperty)
        find_cypher = """
        MATCH (p:ObjectProperty {uri: $uri})
        RETURN p.uri AS rel_type
        """
        res = self.run_custom_query(find_cypher, {"uri": object_property_uri})
        if not res:
            return False

        rel_type = res[0].get("rel_type")
        if rel_type:
            # тип отношения нельзя передать параметром, поэтому вставляем в запрос (экранируем `)
            rel_type_safe = rel_type.replace("`", "``")
            delete_rels = f"""
            MATCH ()-[r:`{rel_type_safe}`]-()
            DELETE r
            """
            self.run_custom_query(delete_rels)

        # удаляем сам узел ObjectProperty
        delete_prop = """
        MATCH (p:ObjectProperty {uri: $uri})
        DETACH DELETE p
        RETURN COUNT(p) > 0 AS deleted
        """
        res_del = self.run_custom_query(delete_prop, {"uri": object_property_uri})
        return res_del[0]["deleted"] if res_del else False

    def add_class_parent(self, parent_uri: str, target_uri: str):
        """
        Присоединить существующего родителя к классу.
        """
        self.create_arc(target_uri, parent_uri, "SUBCLASS_OF")

    # -----------------------
    # Объекты классов
    # -----------------------
    def get_object(self, object_uri: str) -> Optional[TNode]:
        """
        Получить объект класса.
        """
        cypher = "MATCH (o:Object {uri: $uri}) RETURN o"
        res = self.run_custom_query(cypher, {"uri": object_uri})
        return res[0]["o"] if res else None

    def delete_object(self, object_uri: str) -> bool:
        """
        Удалить объект класса (только если это Object).
        """
        cypher = """
        MATCH (o:Object {uri: $uri})
        DETACH DELETE o
        RETURN COUNT(o) > 0 AS deleted
        """
        res = self.run_custom_query(cypher, {"uri": object_uri})
        return res[0]["deleted"] if res else False

    @staticmethod
    def _normalize_object_uris(value: Any) -> List[str]:
        """
        Привести значения object_properties к списку uri.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            uri = value.get("uri")
            return [uri] if uri else []
        if isinstance(value, (list, tuple, set)):
            uris: List[str] = []
            for item in value:
                if isinstance(item, str):
                    uris.append(item)
                elif isinstance(item, dict) and item.get("uri"):
                    uris.append(item["uri"])
            return uris
        return []

    def _delete_object_property_relations(self, object_uri: str, rel_type: str, direction: int) -> None:
        """
        Удалить связи объекта по типу ObjectProperty (в нужном направлении).
        """
        if not rel_type:
            return
        rel_type_safe = rel_type.replace("`", "``")
        if direction == 1:
            cypher = f"""
            MATCH (o:Object {{uri: $uri}})-[r:`{rel_type_safe}`]->()
            DELETE r
            """
        else:
            cypher = f"""
            MATCH ()-[r:`{rel_type_safe}`]->(o:Object {{uri: $uri}})
            DELETE r
            """
        self.run_custom_query(cypher, {"uri": object_uri})

    def _apply_object_properties(self, object_uri: str, obj_params: List[Dict[str, Any]],
                                 object_properties: Dict[str, Any]) -> None:
        """
        Создать связи объекта по object_properties (без удаления старых).
        """
        if not object_properties:
            return

        obj_by_uri = {p.get("uri"): p for p in obj_params if p.get("uri")}
        obj_by_title: Dict[str, Dict[str, Any]] = {}
        for p in obj_params:
            title = p.get("title")
            if title and title not in obj_by_title:
                obj_by_title[title] = p

        for key, value in object_properties.items():
            param = obj_by_uri.get(key) or obj_by_title.get(key)
            if not param:
                continue

            rel_type = param.get("uri")
            direction = int(param.get("relation_direction", 1))
            for target_uri in self._normalize_object_uris(value):
                if direction == 1:
                    self.create_arc(object_uri, target_uri, rel_type)
                else:
                    self.create_arc(target_uri, object_uri, rel_type)

    def _update_object_properties(self, object_uri: str, obj_params: List[Dict[str, Any]],
                                  object_properties: Dict[str, Any]) -> None:
        """
        Обновить связи объекта по object_properties (сначала удалить старые, потом создать новые).
        """
        if not object_properties:
            return

        obj_by_uri = {p.get("uri"): p for p in obj_params if p.get("uri")}
        obj_by_title: Dict[str, Dict[str, Any]] = {}
        for p in obj_params:
            title = p.get("title")
            if title and title not in obj_by_title:
                obj_by_title[title] = p

        for key, value in object_properties.items():
            param = obj_by_uri.get(key) or obj_by_title.get(key)
            if not param:
                continue

            rel_type = param.get("uri")
            direction = int(param.get("relation_direction", 1))
            # заменяем связи по этому свойству
            self._delete_object_property_relations(object_uri, rel_type, direction)

            for target_uri in self._normalize_object_uris(value):
                if direction == 1:
                    self.create_arc(object_uri, target_uri, rel_type)
                else:
                    self.create_arc(target_uri, object_uri, rel_type)

    def create_object(self, class_uri: str, title: str, description: str, properties: Optional[Dict[str, Any]] = None,
                      object_properties: Optional[Dict[str, Any]] = None,
                      ) -> TNode:
        """
        Создать объект класса с datatype и object свойствами.
        """
        properties = properties or {}
        object_properties = object_properties or {}

        signature = self.collect_signature(class_uri)
        dp_by_uri = {p.get("uri"): p.get("title") for p in signature.get("params", []) if
                     p.get("uri") and p.get("title")}
        allowed_titles = {p.get("title") for p in signature.get("params", []) if p.get("title")}

        node_props = {"title": title, "description": description, "class_uri": class_uri}
        for key, value in properties.items():
            if key in allowed_titles:
                node_props[key] = value
            elif key in dp_by_uri:
                node_props[dp_by_uri[key]] = value

        obj = self.create_node(node_props, labels=["Object"])
        self.create_arc(obj["uri"], class_uri, "rdf:type")

        # Create object relations based on provided object_properties
        self._apply_object_properties(obj["uri"], signature.get("obj_params", []), object_properties)
        return obj

    def update_object(
            self,
            object_uri: str,
            title: str,
            description: str,
            properties: Optional[Dict[str, Any]] = None,
            object_properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[TNode]:
        """
    Обновить объект класса с datatype и object свойствами.
    """
        obj = self.get_object(object_uri)
        if not obj:
            return None

        class_uri = obj.get("class_uri")
        signature = self.collect_signature(class_uri) if class_uri else {"params": [], "obj_params": []}
        dp_by_uri = {p.get("uri"): p.get("title") for p in signature.get("params", []) if
                     p.get("uri") and p.get("title")}
        allowed_titles = {p.get("title") for p in signature.get("params", []) if p.get("title")}

        properties = properties or {}
        update_props = {"title": title, "description": description}
        for key, value in properties.items():
            if key in allowed_titles:
                update_props[key] = value
            elif key in dp_by_uri:
                update_props[dp_by_uri[key]] = value

        updated = self.update_node(object_uri, update_props)

        if object_properties is not None:
            self._update_object_properties(object_uri, signature.get("obj_params", []), object_properties)

        return updated

    # -----------------------
    # Сигнатуры
    # -----------------------
    def collect_signature(self, class_uri: str) -> Dict[str, Any]:
        """
        Collect class signature (DatatypeProperty and ObjectProperty), including parents.
        Returns:
        {
          params: [{title, uri}, ...],
          obj_params: [{title, uri, target_class_uri, relation_direction}, ...]
        }
        """
        cypher = """
        MATCH (c:Class {uri: $uri})
        OPTIONAL MATCH (c)-[:SUBCLASS_OF*0..]->(cls:Class)
        WITH collect(DISTINCT cls) AS classes
        UNWIND classes AS cls
        OPTIONAL MATCH (cls)<-[:DOMAIN]-(dp:DatatypeProperty)
        OPTIONAL MATCH (cls)<-[:DOMAIN]-(op1:ObjectProperty)-[:RANGE]->(rc1:Class)
        OPTIONAL MATCH (cls)<-[:RANGE]-(op2:ObjectProperty)-[:DOMAIN]->(rc2:Class)

        RETURN
          collect(DISTINCT dp) AS datatype_props,
          collect(DISTINCT {prop: op1, target: rc1, direction: 1}) AS obj_props_pos,
          collect(DISTINCT {prop: op2, target: rc2, direction: -1}) AS obj_props_neg
        """
        res = self.run_custom_query(cypher, {"uri": class_uri})
        if not res:
            return {"params": [], "obj_params": []}

        row = res[0]
        # DatatypeProperty
        params = [{"title": dp.get("title"), "uri": dp.get("uri")} for dp in row.get("datatype_props", []) if dp]

        # ObjectProperty direction=1
        obj_params_pos = []
        for item in row.get("obj_props_pos", []):
            prop = item.get("prop")
            target = item.get("target")
            if prop and target:
                obj_params_pos.append({
                    "title": prop.get("title"),
                    "uri": prop.get("uri"),
                    "target_class_uri": target.get("uri"),
                    "relation_direction": 1
                })

        # ObjectProperty direction=-1
        obj_params_neg = []
        for item in row.get("obj_props_neg", []):
            prop = item.get("prop")
            target = item.get("target")
            if prop and target:
                obj_params_neg.append({
                    "title": prop.get("title"),
                    "uri": prop.get("uri"),
                    "target_class_uri": target.get("uri"),
                    "relation_direction": -1
                })

        return {
            "params": params,
            "obj_params": obj_params_pos + obj_params_neg
        }
