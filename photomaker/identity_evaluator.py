import torch
import torch.nn.functional as F


def evaluate_identities_dynamic(face_embeds, ref_embeds):
    """
    face_embeds: [num_faces, 512]
    ref_embeds:  [num_identities, 512]
    """

    face_embeds = F.normalize(face_embeds, dim=-1)
    ref_embeds = F.normalize(ref_embeds, dim=-1)

    sim_matrix = torch.matmul(face_embeds, ref_embeds.T)

    print("\n=== Cosine Similarity Matrix ===")
    print(sim_matrix)

    assignments = []

    for face_idx in range(sim_matrix.shape[0]):
        sims = sim_matrix[face_idx]
        best_id = torch.argmax(sims).item()
        best_score = sims[best_id].item()

        assignments.append((face_idx, best_id, best_score))

        print(
            f"Face {face_idx} → Assigned ID {best_id} "
            f"(Cosine: {best_score:.4f})"
        )

    return assignments, sim_matrix
