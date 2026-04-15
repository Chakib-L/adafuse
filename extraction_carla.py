import carla
import os
import random
import time
import json

# --- CONFIGURATION DE LA SIMULATION ---
BASE_DIR = "C:/Users/cress/Desktop/Carla_Single_Dataset"
FRAMES_A_SIMULER = 200   # La durée de votre simulation
NB_TESLAS_LIDAR = 10      # Nombre de Tesla avec capteur
TRAFIC_FOND = 50         # Nombre de voitures "figurantes"

# Fonction d'enregistrement LiDAR + Trajectoire CSV
def creer_fonction_sauvegarde(dossier):
    fichier_csv = f"{dossier}/trajectoire.csv"
    with open(fichier_csv, "w") as f:
        f.write("frame,x,y,z,pitch,yaw,roll\n")
        
    def save_lidar_data(data):
        # Sauvegarde du nuage de points SÉMANTIQUE (avec ID et Tag)
        data.save_to_disk(f"{dossier}/%06d.ply" % data.frame)
        
        # Sauvegarde de la position globale du LiDAR
        t = data.transform
        with open(fichier_csv, "a") as f:
            f.write(f"{data.frame},{t.location.x},{t.location.y},{t.location.z},{t.rotation.pitch},{t.rotation.yaw},{t.rotation.roll}\n")
            
    return save_lidar_data

def main():
    actors_list = []
    lidars_list = []
    
    # 1. Préparation des dossiers
    os.makedirs(BASE_DIR, exist_ok=True)
    gt_folder = f"{BASE_DIR}/ground_truth"
    os.makedirs(gt_folder, exist_ok=True)
    
    try:
        print("1. Connexion au serveur CARLA...")
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(20.0)
        
        # On charge Town01 à neuf pour éviter les bugs de collisions passés
        print("2. Chargement du monde...")
        world = client.load_world('Town01')
        time.sleep(2)

        # Mode Synchrone
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        # 3. Préparation du LiDAR Sémantique et des Tesla
        tesla_bp = bp_lib.find('vehicle.tesla.model3')
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast_semantic') # Capteur enrichi !
        lidar_bp.set_attribute('channels', '16')
        lidar_bp.set_attribute('points_per_second', '20000')
        lidar_bp.set_attribute('range', '30')

        print("3. Apparition des véhicules...")
        teslas = []
        for i in range(NB_TESLAS_LIDAR):
            v = None
            while v is None and spawn_points:
                v = world.try_spawn_actor(tesla_bp, spawn_points.pop(0))
            
            if v:
                actors_list.append(v)
                v.set_autopilot(True, traffic_manager.get_port())
                traffic_manager.ignore_lights_percentage(v, 100.0)
                teslas.append(v)
                
                # Attachement du LiDAR
                dossier_lidar = f"{BASE_DIR}/Tesla_{i+1}"
                os.makedirs(dossier_lidar, exist_ok=True)
                
                l_transform = carla.Transform(carla.Location(z=2.4))
                lidar = world.spawn_actor(lidar_bp, l_transform, attach_to=v)
                actors_list.append(lidar)
                lidars_list.append(lidar)
                lidar.listen(creer_fonction_sauvegarde(dossier_lidar))

        # Trafic de fond
        autos_bp = [bp for bp in bp_lib.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 4]
        for _ in range(TRAFIC_FOND):
            if not spawn_points: break
            npc = world.try_spawn_actor(random.choice(autos_bp), spawn_points.pop(0))
            if npc:
                actors_list.append(npc)
                npc.set_autopilot(True, traffic_manager.get_port())

        # 4. LA SIMULATION (Extraction des données)
        print("\n=== DÉBUT DE L'EXTRACTION ===")
        for f in range(FRAMES_A_SIMULER):
            world.tick()
            frame_id = world.get_snapshot().frame
            
            # --- EXTRACTION DE LA VÉRITÉ TERRAIN (Boîtes 3D) ---
            objets_gt = []
            for vehicule in world.get_actors().filter('vehicle.*'):
                trans = vehicule.get_transform()
                bbox = vehicule.bounding_box
                
                objets_gt.append({
                    "id_instance": vehicule.id,
                    "classe": vehicule.type_id,
                    "position_globale": {
                        "x": trans.location.x, "y": trans.location.y, "z": trans.location.z
                    },
                    "rotation": {
                        "pitch": trans.rotation.pitch, "yaw": trans.rotation.yaw, "roll": trans.rotation.roll
                    },
                    "bounding_box_3d": {
                        "extent": { "x": bbox.extent.x, "y": bbox.extent.y, "z": bbox.extent.z },
                        "centre_local": { "x": bbox.location.x, "y": bbox.location.y, "z": bbox.location.z }
                    }
                })
            
            with open(f"{gt_folder}/%06d.json" % frame_id, "w") as f_json:
                json.dump({"frame": frame_id, "objets": objets_gt}, f_json, indent=4)
            
            # --- CAMÉRA SUIVEUSE ---
            if teslas and teslas[0].is_alive:
                t = teslas[0].get_transform()
                v_dir = t.get_forward_vector()
                t.location.x -= v_dir.x * 10; t.location.y -= v_dir.y * 10; t.location.z += 4
                t.rotation.pitch = -20
                world.get_spectator().set_transform(t)
            
            if f % 20 == 0:
                print(f"--> Frame {f}/{FRAMES_A_SIMULER} enregistrée...")

        print("\n✅ SIMULATION TERMINÉE AVEC SUCCÈS !")

    except Exception as e:
        print(f"⚠️ Erreur durant la simulation : {e}")
    
    finally:
        print("🧹 Nettoyage du serveur CARLA...")
        # Arrêt propre des capteurs
        for capteur in lidars_list:
            if capteur.is_listening:
                capteur.stop()
        
        # Remise en asynchrone
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        time.sleep(1.0)
        # Destruction propre des acteurs
        client.apply_batch([carla.command.DestroyActor(x) for x in actors_list])
        print("Terminé. Le dataset est disponible sur votre bureau !")

if __name__ == '__main__':
    main()