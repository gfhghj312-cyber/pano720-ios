import SwiftUI
import Observation

@main
struct Pano720App: App {
    @State private var store = PanoramaStore()

    var body: some Scene {
        WindowGroup {
            HomeView()
                .environment(store)
        }
    }
}
