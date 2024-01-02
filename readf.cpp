struct Data {
    // mc particle
    vector<int> mcparts_trackID;
    vector<int> mcparts_parentID;
    vector<int> mcparts_pdgID;
    vector<float> mcparts_mass;
    vector<float> mcparts_charge;
    vector<float> mcparts_vertex_x;
    vector<float> mcparts_vertex_y; 
    vector<float> mcparts_vertex_z;
    vector<float> mcparts_momentum_x;
    vector<float> mcparts_momentum_y;
    vector<float> mcparts_momentum_z;

    // fit hits
    vector<int> hit_cellCode;
    vector<int> hit_trackID;
    vector<float> hit_posX;
    vector<float> hit_posY;
    vector<float> hit_posZ;
    vector<float> hit_edep;
    vector<float> hit_pdgID;
};
Data readFile(TString f_mc){

    TChain *ch_mc = new TChain("events");
    ch_mc->Add(f_mc);
    TTreeReader reader_mc(ch_mc);
    Data data(reader_mc);
    //mc particle
    TTreeReaderArray<int> mcparts_trackID(reader_mc, "mcparts.trackID");
    TTreeReaderArray<int> mcparts_parentID(reader_mc, "mcparts.parentID");
    TTreeReaderArray<int> mcparts_pdgID(reader_mc, "mcparts.pdgID");
    TTreeReaderArray<float> mcparts_mass(reader_mc, "mcparts.mass");
    TTreeReaderArray<float> mcparts_vertex_x(reader_mc, "mcparts.vertex.x");
    TTreeReaderArray<float> mcparts_vertex_y(reader_mc, "mcparts.vertex.y");
    TTreeReaderArray<float> mcparts_vertex_z(reader_mc, "mcparts.vertex.z");
    TTreeReaderArray<float> mcparts_momentum_x(reader_mc, "mcparts.momentum.x");
    TTreeReaderArray<float> mcparts_momentum_y(reader_mc, "mcparts.momentum.y");
    TTreeReaderArray<float> mcparts_momentum_z(reader_mc, "mcparts.momentum.z");

   //fit hits
    TTreeReaderArray<int>   hit_cellCode(reader_mc, "fithits.cellCode");
    TTreeReaderArray<int>   hit_trackID(reader_mc, "fithits.trackID");
    TTreeReaderArray<float> hit_posX(reader_mc, "fithits.pos.x");
    TTreeReaderArray<float> hit_posY(reader_mc, "fithits.pos.y");
    TTreeReaderArray<float> hit_posZ(reader_mc, "fithits.pos.z");
    TTreeReaderArray<float> hit_edep(reader_mc, "fithits.edep");
    TTreeReaderArray<float> hit_pdgID(reader_mc, "fithits.pdgID");
     while (reader_mc.Next()) {
        data.mcparts_trackID.insert(data.mcparts_trackID.end(), mcparts_trackID.begin(), mcparts_trackID.end());
     }
     return data;  
}
int main() {
    Data data = readFile("gamma-top10cmx10cmVert-sim.43265645.0.root");
    int Count = 0;
    int nentries = 0;
    float maxedep = *max_element(data.hit_edep.begin(), data.hit_edep.end());
    float minedep = *min_element(data.hit_edep.begin(), data.hit_edep.end());
    float thresholdmax = 0.9 * maxedep;
    float thresholdmin = 1.1 * minedep;
    if (hit_pdgID == 22 && hit_trackID >= 3) {
        for (float edep : hit_edep) {
                if (thresholdmin < edep && edep < thresholdmax) {
                    Count++;
                }
            }
        }
}