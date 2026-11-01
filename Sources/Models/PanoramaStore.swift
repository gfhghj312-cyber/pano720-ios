import Foundation
import UIKit
import Observation

struct PanoramaRecord: Codable, Identifiable, Hashable {
    var id: UUID
    var title: String
    var createdAt: Date
    var fileName: String        // 等距圆柱原图
    var thumbName: String       // 封面
    var width: Int
    var height: Int
    var coverage: Double
    var shotCount: Int

    var aspect: CGFloat { height > 0 ? CGFloat(width) / CGFloat(height) : 2 }
}

@Observable
final class PanoramaStore {

    private(set) var items: [PanoramaRecord] = []
    let root: URL
    private let indexURL: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        root = docs.appendingPathComponent("Panoramas", isDirectory: true)
        indexURL = root.appendingPathComponent("index.json")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        load()
    }

    func imageURL(for record: PanoramaRecord) -> URL {
        root.appendingPathComponent(record.fileName)
    }

    func thumbURL(for record: PanoramaRecord) -> URL {
        root.appendingPathComponent(record.thumbName)
    }

    func image(for record: PanoramaRecord) -> UIImage? {
        UIImage(contentsOfFile: imageURL(for: record).path)
    }

    func thumbnail(for record: PanoramaRecord) -> UIImage? {
        UIImage(contentsOfFile: thumbURL(for: record).path)
    }

    @discardableResult
    func save(_ panorama: StitchedPanorama, shotCount: Int, title: String? = nil) -> PanoramaRecord? {
        let id = UUID()
        let imageName = "\(id.uuidString).jpg"
        let thumbName = "\(id.uuidString)-thumb.jpg"
        do {
            try panorama.jpeg.write(to: root.appendingPathComponent(imageName), options: .atomic)
            try panorama.thumbnail.write(to: root.appendingPathComponent(thumbName), options: .atomic)
        } catch {
            return nil
        }

        let formatter = DateFormatter()
        formatter.dateFormat = "MM-dd HH:mm"
        let record = PanoramaRecord(id: id,
                                    title: title ?? "全景 \(formatter.string(from: Date()))",
                                    createdAt: Date(),
                                    fileName: imageName,
                                    thumbName: thumbName,
                                    width: panorama.width,
                                    height: panorama.height,
                                    coverage: panorama.coverage,
                                    shotCount: shotCount)
        items.insert(record, at: 0)
        persist()
        return record
    }

    func delete(_ record: PanoramaRecord) {
        try? FileManager.default.removeItem(at: imageURL(for: record))
        try? FileManager.default.removeItem(at: thumbURL(for: record))
        items.removeAll { $0.id == record.id }
        persist()
    }

    // MARK: - 持久化

    private func load() {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: indexURL),
              let list = try? decoder.decode([PanoramaRecord].self, from: data) else {
            items = []
            return
        }
        items = list
    }

    private func persist() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted]
        guard let data = try? encoder.encode(items) else { return }
        try? data.write(to: indexURL, options: .atomic)
    }
}
